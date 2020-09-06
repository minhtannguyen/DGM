import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd

import numpy as np


cfg = {
    'ConvSmallMNIST': [[32, 'M', 64, 64, 'M', 128, 'A'], ['full', 'valid', 'full', 'valid'], [5, 3, 3, 3]],
    'ConvSmallMNIST256': [[32, 'M', 64, 64, 'M', 256, 'A'], ['full', 'valid', 'full', 'valid'], [5, 3, 3, 3]]
}

class UpsampleLayer(nn.HybridBlock):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, size=2, scale=1., **kwargs):
        super(UpsampleLayer, self).__init__(**kwargs)
        self._size=size
        self._scale=scale

    def hybrid_forward(self, F, x):
        x = self._scale * x
        x = F.repeat(x, repeats=self._size, axis=2)
        x = F.repeat(x, repeats=self._size, axis=3)
        return x

# Reshape layer
class Reshape(nn.HybridBlock):
    """
    Flatten the output of the convolutional layer
    Parameters
    ----------
    Input shape: (N, C * W * H)
    Output shape: (N, C, W, H)
    """
    def __init__(self, shape, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self._shape = shape

    def hybrid_forward(self, F, x):
        return F.reshape(x, (x.shape[0], self._shape[0], self._shape[1], self._shape[2]))
    
class BiasAdder(nn.HybridBlock):
    """
    Add a bias into the input
    """
    def __init__ (self, channels, **kwargs):
        super(BiasAdder, self).__init__(**kwargs)
        with self.name_scope():
            self.bias = self.params.get('bias', shape=(1,channels,1,1))

    def hybrid_forward(self, F, x, bias):
        with x.context:
            activation = x + bias

        return activation
    
class InstanceNorm(nn.HybridBlock):
    r"""
    Applies instance normalization to the n-dimensional input array.
    This operator takes an n-dimensional input array where (n>2) and normalizes
    the input using the following formula:
    .. math::
      \bar{C} = \{i \mid i \neq 0, i \neq axis\}
      out = \frac{x - mean[data, \bar{C}]}{ \sqrt{Var[data, \bar{C}]} + \epsilon}
       * gamma + beta
    Parameters
    ----------
    axis : int, default 1
        The axis that will be excluded in the normalization process. This is typically the channels
        (C) axis. For instance, after a `Conv2D` layer with `layout='NCHW'`,
        set `axis=1` in `InstanceNorm`. If `layout='NHWC'`, then set `axis=3`. Data will be
        normalized along axes excluding the first axis and the axis given.
    epsilon: float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    center: bool, default True
        If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: bool, default True
        If True, multiply by `gamma`. If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    beta_initializer: str or `Initializer`, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer: str or `Initializer`, default 'ones'
        Initializer for the gamma weight.
    in_channels : int, default 0
        Number of channels (feature maps) in input data. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    Inputs:
        - **data**: input tensor with arbitrary shape.
    Outputs:
        - **out**: output tensor with the same shape as `data`.
    References
    ----------
        `Instance Normalization: The Missing Ingredient for Fast Stylization
        <https://arxiv.org/abs/1607.08022>`_
    Examples
    --------
    >>> # Input of shape (2,1,2)
    >>> x = mx.nd.array([[[ 1.1,  2.2]],
    ...                 [[ 3.3,  4.4]]])
    >>> # Instance normalization is calculated with the above formula
    >>> layer = InstanceNorm()
    >>> layer.initialize(ctx=mx.cpu(0))
    >>> layer(x)
    [[[-0.99998355  0.99998331]]
     [[-0.99998319  0.99998361]]]
    <NDArray 2x1x2 @cpu(0)>
    """
    def __init__(self, axis=1, epsilon=1e-5, center=True, scale=False,
                 beta_initializer='zeros', gamma_initializer='ones',
                 in_channels=0, **kwargs):
        super(InstanceNorm, self).__init__(**kwargs)
        self._kwargs = {'eps': epsilon, 'axis': axis, 'center': center, 'scale': scale}
        self._axis = axis
        self._epsilon = epsilon
        self.gamma = self.params.get('gamma', grad_req='write' if scale else 'null',
                                     shape=(in_channels,), init=gamma_initializer,
                                     allow_deferred_init=True)
        self.beta = self.params.get('beta', grad_req='write' if center else 'null',
                                    shape=(in_channels,), init=beta_initializer,
                                    allow_deferred_init=True)

    def hybrid_forward(self, F, x, gamma, beta):
        if self._axis == 1:
            return F.InstanceNorm(x, gamma, beta,
                                  name='fwd', eps=self._epsilon)
        x = x.swapaxes(1, self._axis)
        return F.InstanceNorm(x, gamma, beta, name='fwd',
                              eps=self._epsilon).swapaxes(1, self._axis)

    def __repr__(self):
        s = '{name}({content}'
        in_channels = self.gamma.shape[0]
        s += ', in_channels={0}'.format(in_channels)
        s += ')'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))

class VGG_DRM(nn.HybridBlock):
    def __init__(self, vgg_name, batch_size, num_class, use_bias=False, use_bn=False, do_topdown=False, do_countpath=False, do_pn=False, relu_td=False, do_nn=False):
        super(VGG_DRM, self).__init__()
        self.num_class = num_class
        self.do_topdown = do_topdown
        self.do_countpath = do_countpath
        self.do_pn = do_pn
        self.relu_td = relu_td
        self.do_nn = do_nn
        self.use_bn = use_bn
        self.use_bias = use_bias
        self.batch_size = batch_size
        self.features, layers_drm, layers_drm_cp = self._make_layers(cfg[vgg_name], use_bias, use_bn, self.do_topdown, self.do_countpath)
        with self.name_scope():
            self.classifier = nn.HybridSequential(prefix='classifier_')
            conv_layer = nn.Conv2D(in_channels=cfg[vgg_name][0][-2], channels=self.num_class, kernel_size=(1, 1), use_bias=True)
            self.classifier.add(conv_layer)
            self.classifier.add(nn.Flatten())

        if self.do_topdown:
            layers_drm += [nn.Conv2DTranspose(channels=cfg[vgg_name][0][-2], in_channels=self.num_class, kernel_size=(1,1), strides=(1, 1),
                                              use_bias=False, params=conv_layer.params),
                           Reshape(shape=(self.num_class, 1, 1))]
            with self.name_scope():
                self.drm = nn.HybridSequential(prefix='drmtd_')
                for block in layers_drm[::-1]:
                    self.drm.add(block)
            if self.do_pn:
                with self.name_scope():
                    self.insnorms = nn.HybridSequential(prefix='instancenorm_')
                    count_bn = 0
                    for i in range(len(self.drm._children)):
                        if (self.drm._children[i].name.find('batchnorm') != -1):
                            count_bn = count_bn + 1
                    for i in range(count_bn - 2):
                        self.insnorms.add(InstanceNorm())
                with self.name_scope():
                    self.insnorms_fw = nn.HybridSequential(prefix='instancenormfw_')
                    for i in range(len(self.features._children)):
                        if (self.features._children[i].name.find('batchnorm') != -1):
                            self.insnorms_fw.add(InstanceNorm())

        if self.do_countpath:
            layers_drm_cp += [nn.Conv2DTranspose(channels=cfg[vgg_name][0][-2], in_channels=self.num_class, kernel_size=(1, 1), strides=(1, 1),
                                              use_bias=False),
                              Reshape(shape=(self.num_class, 1, 1))]
            with self.name_scope():
                self.drm_cp = nn.HybridSequential(prefix='drmcp_')
                for block in layers_drm_cp[::-1]:
                    self.drm_cp.add(block)

    def hybrid_forward(self, F, x, y=None):
        ahat = []; that = []; bfw = []; apn = []; meanfw = []; varfw = []
        xbias = F.zeros((1, x.shape[1], x.shape[2], x.shape[3]), ctx=x.context) if self.do_pn else []
        insnormfw_indx = 0

        if self.do_topdown or self.do_countpath:
            for layer in self.features._children:
                if layer.name.find('pool') != -1 and not layer.name.find('avg') != -1:
                    that.append((x-F.repeat(F.repeat(layer(x), repeats=2, axis=2), repeats=2, axis=3)).__ge__(0))
                    x = layer(x)
                    if self.do_pn:
                        xbias = layer(xbias)
                else:
                    x = layer(x)
                    if self.do_pn:
                        if layer.name.find('batchnorm') != -1:
                            xbias = self.insnorms_fw[insnormfw_indx](xbias)
                            insnormfw_indx += 1
                        else:
                            xbias = layer(xbias)
                    if layer.name.find('relu') != -1:
                        ahat.append(x.__gt__(0) + (x.__le__(0))*0.1)
                        if self.do_pn:
                            apn.append(xbias.__gt__(0) + (xbias.__le__(0))*0.1)
                    
                    if self.use_bn:
                        if layer.name.find('conv') != -1:
                            meanfw.append(F.mean(x, axis=1, exclude=True))
                            varfw.append(F.mean((x - F.mean(x, axis=1, exclude=True, keepdims=True))**2, axis=1, exclude=True))
                        if self.use_bias:
                            if layer.name.find('biasadder') != -1:
                                bfw.append(layer.bias)
                        else:
                            if layer.name.find('batchnorm') != -1:
                                bfw.append(layer.beta)
                    else:
                        if self.use_bias:
                            if layer.name.find('conv') != -1:
                                bfw.append(layer.bias)

            ahat = ahat[::-1]
            that = that[::-1]
            bfw = bfw[::-1]
            apn = apn[::-1]
            meanfw = meanfw[::-1]
            varfw = varfw[::-1]
        else:
            x =  self.features(x)

        z = self.classifier(x)
        zbias = self.classifier(xbias)
        
        cinput = y if y is not None else F.argmax(z.detach(), axis=1)
        cpn = F.argmax(zbias.detach(), axis=1)

        bias_all = self.classifier[0].bias.data()
        lnpicinput = F.take(bias_all, cinput)
        lnpicpn = F.take(bias_all, cpn)

        if self.do_topdown:
            xhat, _, loss_pn, loss_nn = self.topdown(F, self.drm, F.one_hot(y, self.num_class), ahat, that, bfw, F.ones((1, z.shape[1]), ctx=z.context), apn, meanfw, varfw, lnpicinput, lnpicpn) if y is not None \
                else self.topdown(F, self.drm, F.one_hot(F.argmax(z.detach(), axis=1), self.num_class), ahat, that, bfw, F.ones((1, z.shape[1]), ctx=z.context), apn, meanfw, varfw, lnpicinput, lnpicpn)
        else:
            xhat = None
            loss_pn = None
            loss_nn = None

        if self.do_countpath:
            xpath, _, loss_pn, loss_nn = self.topdown(F, self.drm_cp, F.one_hot(y, self.num_class), ahat, that, bfw, F.ones((1, z.shape[1]), ctx=z.context), apn, meanfw, varfw, lnpicinput, lnpicpn) if y is not None \
                else self.topdown(F, self.drm_cp, F.one_hot(F.argmax(z.detach(), axis=1), self.num_class), ahat, that, bfw, F.ones((1, z.shape[1]), ctx=z.context), apn, meanfw, varfw, lnpicinput, lnpicpn)

        else:
            xpath = None
            
        return [z, xhat, xpath, loss_pn, loss_nn]

    def _make_layers(self, cfg, use_bias, use_bn, do_topdown, do_countpath):
        layers = []
        layers_drm = []
        layers_drm_cp = []
        in_channels = 1
        
        conv_indx = 0

        for i, x in enumerate(cfg[0]):
            if x == 'M':
                layers += [nn.MaxPool2D(pool_size=2, strides=2)]
                if do_topdown:
                    if use_bn:
                        layers_drm += [UpsampleLayer(size=2, scale=1.), nn.BatchNorm()]
                    else:
                        layers_drm += [UpsampleLayer(size=2, scale=1.)]
                        
                if do_countpath:
                    if use_bn:
                        layers_drm_cp += [UpsampleLayer(size=2, scale=1.), nn.BatchNorm()]
                    else:
                        layers_drm_cp += [UpsampleLayer(size=2, scale=1.)]
            elif x == 'A':
                layers += [nn.GlobalAvgPool2D(prefix='avg_')]
                if do_topdown:
                    if use_bn:
                        layers_drm += [UpsampleLayer(size=6, scale=1./36., prefix='avg_'), nn.BatchNorm()]
                    else:
                        layers_drm += [UpsampleLayer(size=6, scale=1./36., prefix='avg_')]
                        
                if do_countpath:
                    if use_bn:
                        layers_drm_cp += [UpsampleLayer(size=6, scale=1./36., prefix='avg_'), nn.BatchNorm()]
                    else:
                        layers_drm_cp += [UpsampleLayer(size=6, scale=1./36., prefix='avg_')]
            else:
                if cfg[1][conv_indx] == 'full':
                    pad_val = cfg[2][conv_indx] - 1
                elif cfg[1][conv_indx] == 'half':
                    pad_val = (cfg[2][conv_indx] - 1)/2
                else:
                    pad_val = 0
                    
                padding_fw = (pad_val, pad_val) 
                padding_bw = (pad_val, pad_val)
                conv_size = cfg[2][conv_indx]
                if use_bn:
                    conv_layer = nn.Conv2D(in_channels=in_channels, channels=x, kernel_size=(conv_size, conv_size), padding=padding_fw, use_bias=False)
                    if use_bias:
                        layers += [conv_layer,
                                   nn.BatchNorm(),
                                   BiasAdder(channels=x),
                                   nn.LeakyReLU(alpha=0.1)]
                    else:
                        layers += [conv_layer,
                                   nn.BatchNorm(),
                                   nn.LeakyReLU(alpha=0.1)]
                    if do_topdown:
                        if (cfg[0][i-1] == 'M' or cfg[0][i-1] == 'A') and not i == 0:
                            layers_drm += [nn.Conv2DTranspose(channels=in_channels, in_channels=x, kernel_size=conv_size, strides=(1, 1),
                                                              padding=padding_bw, use_bias=False, params=conv_layer.params)]
                        else:
                            layers_drm += [nn.BatchNorm(), nn.Conv2DTranspose(channels=in_channels, in_channels=x, kernel_size=conv_size, strides=(1, 1),
                                                          padding=padding_bw, use_bias=False, params=conv_layer.params)]
                    if do_countpath:
                        if cfg[0][i-1] == 'M' or cfg[0][i-1] == 'A':
                            layers_drm_cp += [nn.Conv2DTranspose(channels=in_channels, in_channels=x, kernel_size=conv_size, strides=(1, 1),
                                                   padding=padding_bw, use_bias=False)]
                        else:
                            layers_drm_cp += [nn.BatchNorm(), nn.Conv2DTranspose(channels=in_channels, in_channels=x, kernel_size=conv_size, strides=(1, 1),
                                                   padding=padding_bw, use_bias=False)]

                elif use_bias:
                    conv_layer = nn.Conv2D(in_channels=in_channels, channels=x, kernel_size=(conv_size, conv_size), padding=padding_fw, use_bias=True)
                    layers += [conv_layer,
                               nn.LeakyReLU(alpha=0.1)]
                    if do_topdown:
                        layers_drm += [nn.Conv2DTranspose(channels=in_channels, in_channels=x, kernel_size=conv_size, strides=(1, 1),
                                                          padding=padding_bw, use_bias=False, params=conv_layer.params)]
                    if do_countpath:
                        layers_drm_cp += [nn.Conv2DTranspose(channels=in_channels, in_channels=x, kernel_size=conv_size, strides=(1, 1),
                                               padding=padding_bw, use_bias=False)]
                else:
                    conv_layer = nn.Conv2D(in_channels=in_channels, channels=x, kernel_size=(conv_size,conv_size), padding=padding_fw, use_bias=False)
                    layers += [conv_layer,
                               nn.LeakyReLU(alpha=0.1)]
                    if do_topdown:
                        layers_drm += [nn.Conv2DTranspose(channels=in_channels, in_channels=x, kernel_size=conv_size, strides=(1,1),
                                                          padding=padding_bw, use_bias=False, params=conv_layer.params)]
                    if do_countpath:
                        layers_drm_cp += [nn.Conv2DTranspose(channels=in_channels, in_channels=x, kernel_size=conv_size, strides=(1, 1),
                                               padding=padding_bw, use_bias=False)]
                        
                in_channels = x
                conv_indx = conv_indx + 1

        with self.name_scope():
            model = nn.HybridSequential(prefix='features_')
            for block in layers:
                model.add(block)

        return model, layers_drm, layers_drm_cp

    def topdown(self, F, net, xhat, ahat, that, bfw, xpn, apn, meanfw, varfw, lnpicinput, lnpicpn):
        mu = xhat
        mupn = xpn
        loss_nn = F.zeros((self.batch_size,), ctx=mu.context)
        
        loss_pn = F.abs(lnpicinput - lnpicpn)

        ahat_indx = 0; that_indx = 0; meanvar_indx = 0; insnorm_indx = 0
        for i in range(len(net._children)):
            if (net._children[i].name.find('batchnorm') != -1) and (i < len(net._children) - 1):
                if self.do_pn:
                    if (net._children[i+1].name.find('upsamplelayer') != -1):
                        maska = F.Pooling(data=ahat[ahat_indx], kernel=(2,2), pool_type='max', stride=(2,2))
                        maskapn = F.Pooling(data=apn[ahat_indx], kernel=(2,2), pool_type='max', stride=(2,2))
                    else:
                        maska = ahat[ahat_indx]
                        maskapn = apn[ahat_indx]
                    
                    mu_b = bfw[ahat_indx].data().reshape((1, -1, 1, 1)) * mu * maska
                    mupn_b = bfw[ahat_indx].data().reshape((1, -1, 1, 1)) * mupn * maskapn
                    
                    loss_pn_layer = F.mean(F.abs(mu_b - mupn_b), axis=0, exclude=True)
                    loss_pn = loss_pn + loss_pn_layer
                    
            if net._children[i - 1].name.find('upsamplelayer') != -1 and not net._children[i - 1].name.find('avg') != -1:
                mu = mu * that[that_indx]
                if self.do_pn:
                    mupn = mupn * that[that_indx]
                that_indx += 1
                    
            if net._children[i].name.find('conv') != -1 and i > 1:
                if self.do_nn and not self.use_bn:
                    loss_nn = loss_nn + F.mean((F.relu(-mu).reshape((self.batch_size,-1)))**2, axis=1)
                    
                if self.relu_td:
                    mu = F.relu(mu)
                    
                mu = mu * ahat[ahat_indx]
                
                if self.do_pn:
                    mupn = mupn * apn[ahat_indx]

                ahat_indx += 1
                
            mu = net._children[i](mu)
            if (net._children[i].name.find('batchnorm') != -1) and (i < len(net._children) - 1):
                if self.do_pn:
                    if insnorm_indx < len(self.insnorms._children):
                        mupn = self.insnorms._children[insnorm_indx](mupn)
                        insnorm_indx += 1
            else:
                if self.do_pn:
                    mupn = net._children[i](mupn)
            
            if (net._children[i].name.find('conv') != -1) and (i != (len(net._children)-2)):
                if self.do_nn and self.use_bn:
                    loss_nn = loss_nn + 0.5*F.mean(((F.mean(mu, axis=1, exclude=True) - meanfw[meanvar_indx])**2)/varfw[meanvar_indx]) + 0.5*F.mean(F.mean((mu - F.mean(mu, axis=1, exclude=True, keepdims=True))**2, axis=1, exclude=True)/varfw[meanvar_indx]) - 0.5*F.mean(F.log(F.mean((mu - F.mean(mu, axis=1, exclude=True, keepdims=True))**2, axis=1, exclude=True)) - F.log(varfw[meanvar_indx])) - 0.5
                    meanvar_indx += 1
                   
        if self.relu_td:
            mu = F.relu(mu)
            
        return mu, mupn, loss_pn, loss_nn
    
    def render(self, net, mu, a, t, level=0):
        a_indx = 0; t_indx = 0
        for i in range(len(net._children)):
            if net._children[i - 1].name.find('upsamplelayer') != -1 and not net._children[i - 1].name.find('avg') != -1:
                mu = mu * t[t_indx]
                t_indx += 1
            
            if net._children[i].name.find('conv') != -1 and i > 1:
                mu = mu * a[a_indx]
                a_indx += 1
            
            mu = net._children[i](mu)
            if i >= level:
                break
        
        return mu
    
    def renderReLU(self, net, mu, t, b):
        t_indx = 0; b_indx = 0
        for i in range(len(net._children)):
            if net._children[i - 1].name.find('upsamplelayer') != -1 and not net._children[i - 1].name.find('avg') != -1:
                mu = mu * t[t_indx]
                t_indx += 1
            
            if net._children[i].name.find('conv') != -1 and i > 1:
                # bmu = b[b_indx] * mu
                # a = bmu.__gt__(0) + (bmu.__le__(0))*0.1
                # mu = mu * a
                mu = nd.relu(mu)
                b_indx += 1
            
            mu = net._children[i](mu)
        
        return mu
    
    def sample_latents(self, C, H, W, ctx):
        xbias = nd.zeros((1, C, H, W), ctx=ctx)
        insnormfw_indx = 0
        a = []; t = []; b = []
        for layer in self.features._children:
            if layer.name.find('pool') != -1 and not layer.name.find('avg') != -1:
                t.append((xbias-nd.repeat(nd.repeat(layer(xbias), repeats=2, axis=2), repeats=2, axis=3)).__ge__(0))
                xbias = layer(xbias)
            else:
                if layer.name.find('batchnorm') != -1:
                    xbias = self.insnorms_fw[insnormfw_indx](xbias)
                    insnormfw_indx += 1
                else:
                    xbias = layer(xbias)
                if layer.name.find('relu') != -1:
                    a.append(xbias.__gt__(0) + (xbias.__le__(0))*0.1)
                if layer.name.find('biasadder') != -1:
                    b.append(layer.bias.data().reshape((1, -1, 1, 1)))
        
        a = a[::-1]
        t = t[::-1]
        b = b[::-1]
        return a, t, b
    
    def generate(self, y, ctx):
        mu = nd.one_hot(y, self.num_class)
        a, t, b = self.sample_latents(1, 28, 28, ctx)
        mu = self.render(self.drm, mu, a, t)
        # mu = self.renderReLU(self.drm, mu, t, b)
        return mu
    
    def extract_latents(self, x, C, H, W, ctx):
        a = []; t = []; b = []
        for layer in self.features._children:
            if layer.name.find('pool') != -1 and not layer.name.find('avg') != -1:
                t.append((x-nd.repeat(nd.repeat(layer(x), repeats=2, axis=2), repeats=2, axis=3)).__ge__(0))
                x = layer(x)
            else:
                x = layer(x)
                if layer.name.find('relu') != -1:
                    a.append(x.__gt__(0) + (x.__le__(0))*0.1)
                if layer.name.find('biasadder') != -1:
                    b.append(layer.bias.data().reshape((1, -1, 1, 1)))
        
        a = a[::-1]
        t = t[::-1]
        b = b[::-1]
        return a, t, b
    
    def reconstruct(self, x, y, ctx):
        mu = nd.one_hot(y, self.num_class)
        a, t, b = self.extract_latents(x, 1, 28, 28, ctx)
        mu = self.render(self.drm, mu, a, t)
        # mu = self.renderReLU(self.drm, mu, t, b)
        return mu
    
    def reconstruct_at_level(self, x, y, ctx, level):
        mu = nd.one_hot(y, self.num_class)
        a, t, b = self.extract_latents(x, 1, 28, 28, ctx)
        mu = self.render(self.drm, mu, a, t, level)
        # mu = self.renderReLU(self.drm, mu, t, b)
        return mu