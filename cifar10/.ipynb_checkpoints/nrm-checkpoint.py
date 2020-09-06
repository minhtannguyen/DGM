import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd

import numpy as np


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG22': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'AllConv13': [128, 128, 128, 'M', 256, 256, 256, 'M', 512, 256, 128, 'A'],
}

class NRM(nn.HybridBlock):
    def __init__(self, net_name, batch_size, num_class, use_bias=False, use_bn=False, do_topdown=False, do_countpath=False, do_pn=False, relu_td=False, do_nn=False):
        super(NRM, self).__init__()
        self.num_class = num_class
        self.do_topdown = do_topdown
        self.do_countpath = do_countpath
        self.do_pn = do_pn
        self.relu_td = relu_td
        self.do_nn = do_nn
        self.use_bn = use_bn
        self.use_bias = use_bias
        self.batch_size = batch_size
        self.features, layers_drm, layers_drm_cp = self._make_layers(cfg[net_name], use_bias, use_bn, self.do_topdown, self.do_countpath)
        with self.name_scope():
            self.classifier = nn.HybridSequential(prefix='classifier_')
            conv_layer = nn.Conv2D(in_channels=cfg[net_name][-2], channels=self.num_class, kernel_size=(1, 1), use_bias=True)
            self.classifier.add(conv_layer)
            self.classifier.add(nn.Flatten())

        if self.do_topdown:
            layers_drm += [nn.Conv2DTranspose(channels=cfg[net_name][-2], in_channels=self.num_class, kernel_size=(1,1), strides=(1, 1),
                                              use_bias=False, params=conv_layer.params),
                           Reshape(shape=(self.num_class, 1, 1))]
            with self.name_scope():
                self.drm = nn.HybridSequential(prefix='drmtd_')
                for block in layers_drm[::-1]:
                    self.drm.add(block)
            if self.do_pn:
                with self.name_scope():
                    self.insnorms = nn.HybridSequential(prefix='instancenorm_')
                    for i in range(len(self.drm._children)):
                        if (self.drm._children[i].name.find('batchnorm') != -1) and (i < (len(self.drm._children) - 1)):
                            self.insnorms.add(InstanceNorm())
                with self.name_scope():
                    self.insnorms_fw = nn.HybridSequential(prefix='instancenormfw_')
                    for i in range(len(self.features._children)):
                        if (self.features._children[i].name.find('batchnorm') != -1):
                            self.insnorms_fw.add(InstanceNorm())

        if self.do_countpath:
            layers_drm_cp += [nn.Conv2DTranspose(channels=cfg[net_name][-2], in_channels=self.num_class, kernel_size=(1, 1), strides=(1, 1),
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

        if self.do_topdown:
            xhat, _, loss_pn, loss_nn = self.topdown(F, self.drm, F.one_hot(y, self.num_class), ahat, that, bfw, F.ones((1, z.shape[1]), ctx=z.context), apn, meanfw, varfw) if y is not None \
                else self.topdown(F, self.drm, F.one_hot(F.argmax(z.detach(), axis=1), self.num_class), ahat, that, bfw, F.ones((1, z.shape[1]), ctx=z.context), apn, meanfw, varfw)
        else:
            xhat = None
            loss_pn = None
            loss_nn = None

        if self.do_countpath:
            xpath, _, loss_pn, loss_nn = self.topdown(F, self.drm_cp, F.one_hot(y, self.num_class), ahat, that, bfw, F.ones((1, z.shape[1]), ctx=z.context), apn, meanfw, varfw) if y is not None \
                else self.topdown(F, self.drm_cp, F.one_hot(F.argmax(z.detach(), axis=1), self.num_class), ahat, that, bfw, F.ones((1, z.shape[1]), ctx=z.context), apn, meanfw, varfw)

        else:
            xpath = None
            
        return [z, xhat, xpath, loss_pn, loss_nn]

    def _make_layers(self, cfg, use_bias, use_bn, do_topdown, do_countpath):
        layers = []
        layers_drm = []
        layers_drm_cp = []
        in_channels = 3

        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2D(pool_size=2, strides=2), nn.Dropout(0.5)]
                if do_topdown:
                    if use_bn:
                        layers_drm += [UpsampleLayer(size=2, scale=1.), nn.Dropout(0.5), nn.BatchNorm()]
                    else:
                        layers_drm += [UpsampleLayer(size=2, scale=1.), nn.Dropout(0.5)]
                        
                if do_countpath:
                    if use_bn:
                        layers_drm_cp += [UpsampleLayer(size=2, scale=1.), nn.Dropout(0.5), nn.BatchNorm()]
                    else:
                        layers_drm_cp += [UpsampleLayer(size=2, scale=1.), nn.Dropout(0.5)]
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
                padding_fw = (0,0) if x == 512 else (1,1)
                padding_bw = (0,0) if x == 512 else (1,1)
                if use_bn:
                    conv_layer = nn.Conv2D(in_channels=in_channels, channels=x, kernel_size=(3,3), padding=padding_fw, use_bias=False)
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
                        if (cfg[i-1] == 'M' or cfg[i-1] == 'A') and not i == 0:
                            layers_drm += [nn.Conv2DTranspose(channels=in_channels, in_channels=x, kernel_size=3, strides=(1, 1),
                                                              padding=padding_bw, use_bias=False, params=conv_layer.params)]
                        else:
                            layers_drm += [nn.BatchNorm(), nn.Conv2DTranspose(channels=in_channels, in_channels=x, kernel_size=3, strides=(1, 1),
                                                          padding=padding_bw, use_bias=False, params=conv_layer.params)]
                    if do_countpath:
                        if cfg[i-1] == 'M' or cfg[i-1] == 'A':
                            layers_drm_cp += [nn.Conv2DTranspose(channels=in_channels, in_channels=x, kernel_size=3, strides=(1, 1),
                                                   padding=padding_bw, use_bias=False)]
                        else:
                            layers_drm_cp += [nn.BatchNorm(), nn.Conv2DTranspose(channels=in_channels, in_channels=x, kernel_size=3, strides=(1, 1),
                                                   padding=padding_bw, use_bias=False)]

                elif use_bias:
                    conv_layer = nn.Conv2D(in_channels=in_channels, channels=x, kernel_size=(3,3), padding=padding_fw, use_bias=True)
                    layers += [conv_layer,
                               nn.LeakyReLU(alpha=0.1)]
                    if do_topdown:
                        layers_drm += [nn.Conv2DTranspose(channels=in_channels, in_channels=x, kernel_size=3, strides=(1, 1),
                                                          padding=padding_bw, use_bias=False, params=conv_layer.params)]
                    if do_countpath:
                        layers_drm_cp += [nn.Conv2DTranspose(channels=in_channels, in_channels=x, kernel_size=3, strides=(1, 1),
                                               padding=padding_bw, use_bias=False)]
                else:
                    conv_layer = nn.Conv2D(in_channels=in_channels, channels=x, kernel_size=(3,3), padding=padding_fw, use_bias=False)
                    layers += [conv_layer,
                               nn.LeakyReLU(alpha=0.1)]
                    if do_topdown:
                        layers_drm += [nn.Conv2DTranspose(channels=in_channels, in_channels=x, kernel_size=3, strides=(1,1),
                                                          padding=padding_bw, use_bias=False, params=conv_layer.params)]
                    if do_countpath:
                        layers_drm_cp += [nn.Conv2DTranspose(channels=in_channels, in_channels=x, kernel_size=3, strides=(1, 1),
                                               padding=padding_bw, use_bias=False)]
                        
                in_channels = x

        with self.name_scope():
            model = nn.HybridSequential(prefix='features_')
            for block in layers:
                model.add(block)

        return model, layers_drm, layers_drm_cp

    def topdown(self, F, net, xhat, ahat, that, bfw, xpn, apn, meanfw, varfw):
        mu = xhat
        mupn = xpn
        loss_pn = F.zeros((self.batch_size,), ctx=mu.context)
        loss_nn = F.zeros((self.batch_size,), ctx=mu.context)

        ahat_indx = 0; that_indx = 0; meanvar_indx = 0; insnorm_indx = 0
        for i in range(len(net._children)):
            if net._children[i].name.find('conv') != -1 and i > 1:
                if self.do_nn and not self.use_bn:
                    loss_nn = loss_nn + F.mean((F.relu(-mu).reshape((self.batch_size,-1)))**2, axis=1)
                    
                if self.relu_td:
                    mu = F.relu(mu)
                    
                mu = mu * ahat[ahat_indx]
                
                if self.do_pn:
                    mupn = mupn * apn[ahat_indx]
                    mu_b = bfw[ahat_indx].data().reshape((1, -1, 1, 1)) * mu
                    mupn_b = bfw[ahat_indx].data().reshape((1, -1, 1, 1)) * mupn
                    
                    loss_pn_layer = F.mean(F.abs(mu_b - mupn_b), axis=0, exclude=True)
                    loss_pn = loss_pn + loss_pn_layer

                ahat_indx += 1

            if net._children[i - 1].name.find('upsamplelayer') != -1 and not net._children[i - 1].name.find('avg') != -1:
                mu = mu * that[that_indx]
                if self.do_pn:
                    mupn = mupn * that[that_indx]
                that_indx += 1

            mu = net._children[i](mu)
            if (net._children[i].name.find('batchnorm') != -1) and (i < len(net._children) - 1):
                if self.do_pn:
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