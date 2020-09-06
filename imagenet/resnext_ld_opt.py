from __future__ import division

__all__ = ['ResNext', 'Block', 'get_resnext',
           'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_64x4d',
           'se_resnext50_32x4d', 'se_resnext101_32x4d', 'se_resnext101_64x4d']

import os
import math
from mxnet import cpu
from mxnet.gluon import nn

import mxnet as mx
from mxnet import nd

import numpy as np

# Specification
resnext_spec = {'resnext50': [3, 4, 6, 3],
                'resnext101': [3, 4, 23, 3],
                'resnext152': [3, 8, 36, 3]}
resnext_nfilter = [64, 256, 512, 1024]

def compute_kl_gaussian(F, x, meanfw, varfw, meanvar_indx):
    return 0.5*F.mean(((F.mean(x, axis=1, exclude=True) - meanfw[meanvar_indx])**2)/(varfw[meanvar_indx] + 1e-8)) + 0.5*F.mean(F.mean((x - F.mean(x, axis=1, exclude=True, keepdims=True))**2, axis=1, exclude=True)/(varfw[meanvar_indx] + 1e-8)) - 0.5*F.mean(F.log(F.mean((x - F.mean(x, axis=1, exclude=True, keepdims=True))**2, axis=1, exclude=True)+ + 1e-8) - F.log(varfw[meanvar_indx] + 1e-8)) - 0.5

class FirstBlock(nn.HybridBlock):
    r"""First bottle-neck block in the ResNext
    """
    def __init__(self, channels, filter_size, stride, padding, in_channels=0, **kwargs):
        super(FirstBlock, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels, filter_size, stride, padding, in_channels=in_channels))
        self.body.add(nn.BatchNorm())
        self.body.add(BiasAdder(channels=channels))
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.MaxPool2D(pool_size=2, strides=2))
        
        self.insnorms = nn.HybridSequential(prefix='insnorms_')
        for layer in self.body:
            if layer.name.find('batchnorm') != -1:
                self.insnorms.add(InstanceNorm())
    
    def hybrid_forward(self, F, x):
        ahat = []; meanfw = []; varfw = []
        bfw = []
        insnorm_indx = 0
        for layer in self.body:
            if layer.name.find('pool') != -1 and not layer.name.find('avg') != -1:
                that = (x-F.repeat(F.repeat(layer(x), repeats=2, axis=2), repeats=2, axis=3)).__ge__(0)
            if layer.name.find('batchnorm') != -1:
                xinput = F.slice_axis(x, axis=0, begin=0, end=-1)
                xbias = F.slice_axis(x, axis=0, begin=-1, end=None)
                xinput = layer(xinput)
                xbias = self.insnorms[insnorm_indx](xbias)
                insnorm_indx += 1
                x = F.concat(xinput, xbias, dim=0)
            else:
                x = layer(x)
            if layer.name.find('relu') != -1:
                ahat.append(x.__gt__(0))
            if layer.name.find('conv') != -1:
                xinput = F.slice_axis(x, axis=0, begin=0, end=-1)
                meanfw.append(F.mean(xinput, axis=1, exclude=True))
                varfw.append(F.mean((xinput - F.mean(xinput, axis=1, exclude=True, keepdims=True))**2, axis=1, exclude=True))
            if layer.name.find('biasadder') != -1:
                bfw.append(layer.bias)
                
        return x, that, ahat, meanfw, varfw, bfw

class LastTopdownBlock(nn.HybridBlock):
    r"""First topdown block in the DRM
    """
    def __init__(self, block_unit, channels, filter_size, stride, padding, out_channels=0, **kwargs):
        super(LastTopdownBlock, self).__init__(**kwargs)
        output_padding = 1 if stride > 1 else 0
        layers_body = []
        
        self.body = nn.HybridSequential(prefix='')
        # THIS IS MISSING A BATCHNORM
        layers_body += [nn.BatchNorm(), nn.Conv2DTranspose(channels=out_channels, in_channels=channels, kernel_size=filter_size, strides=stride, padding=padding, output_padding=output_padding, use_bias=False, params=block_unit.body[0].params), UpsampleLayer(size=2, scale=1.), nn.BatchNorm()]
        for layer in layers_body[::-1]:
            self.body.add(layer)
        
        self.insnorms = nn.HybridSequential(prefix='insnorms_')
        for layer in self.body:
            if layer.name.find('batchnorm') != -1:
                self.insnorms.add(InstanceNorm())
            
    def hybrid_forward(self, F, x, ahat, that, meanfw, varfw, bfw):
        ahat_indx = 0; meanvar_indx = 0
        insnorm_indx = 0;
        
        shape_x = x.shape
        loss_mm = F.zeros((shape_x[0]-1,), ctx=x.context)
        rpn = F.zeros((shape_x[0]-1,), ctx=x.context)
        
        for i, layer in enumerate(self.body):
            if layer.name.find('conv') != -1:
                x = x * ahat[ahat_indx]
                ahat_indx += 1
                
            if layer.name.find('batchnorm') != -1:
                xcg = F.slice_axis(x, axis=0, begin=0, end=-1)
                xpn = F.slice_axis(x, axis=0, begin=-1, end=None)
                
                if i != (len(self.body._children)-1):
                    bfw_reshape = bfw[ahat_indx].data().reshape((1, -1, 1, 1))
                    rpn_val = F.mean(F.abs(bfw_reshape * (xcg - xpn)), axis=0, exclude=True)
                    rpn = rpn + rpn_val
                    
                    loss_mm_val = compute_kl_gaussian(F, xcg, meanfw, varfw, meanvar_indx)
                    loss_mm = loss_mm + loss_mm_val
                    meanvar_indx += 1
                    
                xcg = layer(xcg)
                
                xpn = self.insnorms[insnorm_indx](xpn)
                insnorm_indx += 1
                
                x = F.concat(xcg, xpn, dim=0)
            else:
                x = layer(x)
                
            if layer.name.find('upsamplelayer') != -1 and not layer.name.find('avg') != -1:
                x = x * that

        return x, loss_mm_val, rpn

class ResNextBlock(nn.HybridBlock):
    r"""Bottleneck Block from `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    """
    def __init__(self, channels, cardinality, bottleneck_width, stride,
                 downsample=False, in_channels=0, **kwargs):
        super(ResNextBlock, self).__init__(**kwargs)
        D = int(math.floor(channels * (bottleneck_width / 64)))
        group_width = cardinality * D

        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(group_width, kernel_size=1, in_channels=in_channels, use_bias=False))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(group_width, kernel_size=3, strides=stride, padding=1, in_channels=group_width,
                                use_bias=False, groups=cardinality))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels * 4, kernel_size=1, in_channels=group_width, use_bias=False))
        self.body.add(nn.BatchNorm())

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels * 4, kernel_size=1, strides=stride, in_channels=in_channels,
                                          use_bias=False))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None
        
        self.insnorms = nn.HybridSequential(prefix='insnorms_')
        for layer in self.body:
            if layer.name.find('batchnorm') != -1:
                self.insnorms.add(InstanceNorm())

        if downsample:
            self.insnormsds = nn.HybridSequential(prefix='insnormsds_')
            for layer in self.downsample:
                if layer.name.find('batchnorm') != -1:
                    self.insnormsds.add(InstanceNorm())
        
        self.biasadder = BiasAdder(channels=channels * 4)

    def hybrid_forward(self, F, x):
        residual = x
        ahat = []; meanfw = []; varfw = []; meanfwds = []; varfwds = []
        bfw = []
        insnorm_indx = 0; insnormds_indx = 0
        
        for layer in self.body:
            if layer.name.find('batchnorm') != -1:
                xinput = F.slice_axis(x, axis=0, begin=0, end=-1)
                xbias = F.slice_axis(x, axis=0, begin=-1, end=None)
                xinput = layer(xinput)
                xbias = self.insnorms[insnorm_indx](xbias)
                insnorm_indx += 1
                x = F.concat(xinput, xbias, dim=0)
            else:
                x = layer(x)
            if layer.name.find('relu') != -1:
                ahat.append(x.__gt__(0))
            if layer.name.find('conv') != -1:
                xinput = F.slice_axis(x, axis=0, begin=0, end=-1)
                meanfw.append(F.mean(xinput, axis=1, exclude=True))
                varfw.append(F.mean((xinput - F.mean(xinput, axis=1, exclude=True, keepdims=True))**2, axis=1, exclude=True))

        if self.downsample:
            for layer in self.downsample:
                if layer.name.find('batchnorm') != -1:
                    xinput = F.slice_axis(residual, axis=0, begin=0, end=-1)
                    xbias = F.slice_axis(residual, axis=0, begin=-1, end=None)
                    xinput = layer(xinput)
                    xbias = self.insnormsds[insnormds_indx](xbias)
                    insnormds_indx += 1
                    residual = F.concat(xinput, xbias, dim=0)
                else:
                    residual = layer(residual)
                
                if layer.name.find('conv') != -1:
                    xinput = F.slice_axis(residual, axis=0, begin=0, end=-1)
                    meanfwds.append(F.mean(xinput, axis=1, exclude=True))
                    varfwds.append(F.mean((xinput - F.mean(xinput, axis=1, exclude=True, keepdims=True))**2, axis=1, exclude=True))

        x = x + residual
        x = self.biasadder(x)
        bfw.append(self.biasadder.bias)
        x = F.Activation(x, act_type='relu')
        ahat.append(x.__gt__(0))
        
        return x, ahat, meanfw, varfw, meanfwds, varfwds, bfw
    
class TopdownBlock(nn.HybridBlock):
    r"""DRM Block for the Bottleneck Block from `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    block_unit: Block instant
        The corresponding Bottleneck Block
    out_channels: int
        Number of channels of the output image
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    stride : int
        Stride size.
    """
    def __init__(self, block_unit, out_channels, channels, cardinality, bottleneck_width, stride, **kwargs):
        super(TopdownBlock, self).__init__(**kwargs)
        D = int(math.floor(channels * (bottleneck_width / 64)))
        group_width = cardinality * D
        output_padding = 1 if stride > 1 else 0

        if block_unit.downsample is not None:
            self.upsample = nn.HybridSequential(prefix='')
            self.upsample.add(nn.BatchNorm())
            self.upsample.add(nn.Conv2DTranspose(channels=out_channels, in_channels=channels * 4, kernel_size=1, strides=stride, use_bias=False, output_padding=output_padding, params=block_unit.downsample[0].params))
        else:
            self.upsample = None
                
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Conv2DTranspose(channels=group_width, in_channels=channels * 4, kernel_size=1, use_bias=False, params=block_unit.body[-2].params))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Conv2DTranspose(channels=group_width, in_channels=group_width, kernel_size=3, strides=stride, padding=1, output_padding=output_padding, use_bias=False, groups=cardinality, params=block_unit.body[-5].params))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Conv2DTranspose(channels=out_channels, in_channels=group_width, kernel_size=1, use_bias=False, params=block_unit.body[-8].params))
        
        self.insnorms = nn.HybridSequential(prefix='insnorms_')
        for layer in self.body:
            if layer.name.find('batchnorm') != -1:
                self.insnorms.add(InstanceNorm())
                
        if self.upsample is not None:
            self.insnormsds = nn.HybridSequential(prefix='insnormsds_')
            for layer in self.upsample:
                if layer.name.find('batchnorm') != -1:
                    self.insnormsds.add(InstanceNorm())         

    def hybrid_forward(self, F, x, ahat, meanfw, varfw, meanfwds, varfwds, bfw):
        ahat_indx = 0; meanvar_indx = 0; meanvards_indx = 0
        insnorm_indx = 0; insnormds_indx = 0
        
        shape_x = x.shape
        loss_mm = F.zeros((shape_x[0]-1,), ctx=x.context)
        
        xcg = F.slice_axis(x, axis=0, begin=0, end=-1)
        xpn = F.slice_axis(x, axis=0, begin=-1, end=None)
        
        # THIS IS MISSING HADAMARD PRODUCT WITH AHAT
        bfw_reshape = bfw[0].data().reshape((1, -1, 1, 1))
        rpn = F.mean(F.abs(bfw_reshape * (xcg - xpn)), axis=0, exclude=True)
        
        residual = x
        
        if self.upsample is not None:
            for layer in self.upsample:
                if layer.name.find('conv') != -1:
                    residual = residual * ahat[ahat_indx]
                    
                if layer.name.find('batchnorm') != -1:
                    residualcg = F.slice_axis(residual, axis=0, begin=0, end=-1)
                    loss_mm_val = compute_kl_gaussian(F, residualcg, meanfwds, varfwds, meanvards_indx)
                    loss_mm = loss_mm + loss_mm_val
                    meanvards_indx += 1
                    residualcg = layer(residualcg)
                    
                    residualpn = F.slice_axis(residual, axis=0, begin=-1, end=None)
                    residualpn = self.insnormsds[insnormds_indx](residualpn)
                    insnormds_indx += 1
                    
                    residual = F.concat(residualcg, residualpn, dim=0)
                else:
                    residual = layer(residual)
                    
        for layer in self.body:
            if layer.name.find('conv') != -1:
                x = x * ahat[ahat_indx]
                ahat_indx += 1
                
            if layer.name.find('batchnorm') != -1:
                xcg = F.slice_axis(x, axis=0, begin=0, end=-1)
                loss_mm_val = compute_kl_gaussian(F, xcg, meanfw, varfw, meanvar_indx)
                loss_mm = loss_mm + loss_mm_val
                meanvar_indx += 1
                xcg = layer(xcg)
                
                xpn = F.slice_axis(x, axis=0, begin=-1, end=None)
                xpn = self.insnorms[insnorm_indx](xpn)
                insnorm_indx += 1
                
                x = F.concat(xcg, xpn, dim=0)
            else:
                x = layer(x)

        x = x + residual
        
        return x, loss_mm_val, rpn
    
class ResNext(nn.HybridBlock):
    r"""ResNext model from
    `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    layers : list of int
        Numbers of layers in each block
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self, resnext_name, cardinality, bottleneck_width,
                 classes=1000, **kwargs):
        super(ResNext, self).__init__(**kwargs)
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        layers = resnext_spec[resnext_name]
        channels = 64
        self.classes = classes
        
        layers_drm = []

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='features_')
            self.features.add(FirstBlock(channels=channels, filter_size=7, stride=2, padding=3, in_channels=3, prefix='firstblock_'))
            layers_drm += [LastTopdownBlock(block_unit=self.features[0], channels=channels, filter_size=7, stride=2, padding=3, out_channels=3, prefix='lastblock_')]

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                resnext_layer = self._make_layer(channels, num_layer, stride, i+1, in_channels=resnext_nfilter[i])
                self.features.add(resnext_layer)
                if i == 0:
                    out_channels = channels
                else:
                    out_channels = channels * 2
                layers_drm += [self._make_drm_layer(resnext_layer, out_channels, channels, num_layer, stride, i+1)]
                channels *= 2
            
            self.features.add(nn.AvgPool2D(7))
            layers_drm += [UpsampleLayer(size=7, scale=1./(7**2), prefix='avg_')]
            
            self.classifier = nn.HybridSequential(prefix='classifier_')
            conv_layer = nn.Conv2D(in_channels=2048, channels=self.classes, kernel_size=(1, 1), use_bias=True)
            self.classifier.add(conv_layer)
            self.classifier.add(nn.Flatten())
            layers_drm += [nn.Conv2DTranspose(channels=2048, in_channels=self.classes, kernel_size=(1,1), strides=(1, 1), use_bias=False, params=conv_layer.params), Reshape(shape=(self.classes, 1, 1))]
            
        with self.name_scope():
            self.drm = nn.HybridSequential(prefix='drmtd_')
            for block in layers_drm[::-1]:
                self.drm.add(block)
    
    # pylint: disable=unused-argument
    def hybrid_forward(self, F, x, y=None):
        ahat = []; meanfw = []; varfw = []; meanfwds = []; varfwds = []; bfw = []
        shape_x = x.shape
        xbias = F.zeros((1, shape_x[1], shape_x[2], shape_x[3]), ctx=x.context)
        x = F.concat(x, xbias, dim=0)
        for layer in self.features:
            if layer.name.find('firstblock') != -1:
                x, that, ahat_val, meanfw_val, varfw_val, bfw_val = layer(x)
                ahat.append(ahat_val)
                meanfw.append(meanfw_val)
                varfw.append(varfw_val)
                bfw.append(bfw_val)
            elif layer.name.find('stage') != -1:
                for sublayer in layer:
                    x, ahat_val, meanfw_val, varfw_val, meanfwds_val, varfwds_val, bfw_val = sublayer(x)
                    ahat.append(ahat_val)
                    meanfw.append(meanfw_val)
                    varfw.append(varfw_val)
                    meanfwds.append(meanfwds_val)
                    varfwds.append(varfwds_val)
                    bfw.append(bfw_val)
            else:
                x = layer(x)
        
        z = self.classifier(x)
        zinput = F.slice_axis(z, axis=0, begin=0, end=-1)
        zbias = F.slice_axis(z, axis=0, begin=-1, end=None)
        
        cinput = y if y is not None else F.argmax(zinput.detach(), axis=1)
        cpn = F.argmax(zbias.detach(), axis=1)

        bias_all = self.classifier[0].bias.data()
        lnpicinput = F.take(bias_all, cinput)
        lnpicpn = F.take(bias_all, cpn)

        mu_val = F.one_hot(cinput, self.classes)
        mupn_val = F.ones((1, self.classes), ctx=z.context)
        mu = F.concat(mu_val, mupn_val, dim=0)
        xhat, loss_mm, rpn = self.topdown(F, self.drm, mu, ahat[::-1], that, meanfw[::-1], varfw[::-1], meanfwds[::-1], varfwds[::-1], bfw[::-1], lnpicinput, lnpicpn)

        return zinput, F.slice_axis(xhat, axis=0, begin=0, end=-1), loss_mm, rpn

    def _make_layer(self, channels, num_layers, stride, stage_index, in_channels):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(ResNextBlock(channels, self.cardinality, self.bottleneck_width,
                            stride, True, in_channels=in_channels, prefix='block1_'))
            for i in range(num_layers-1):
                layer.add(ResNextBlock(channels, self.cardinality, self.bottleneck_width,
                                1, False, in_channels=channels * 4, prefix='block%d_'%(i+2)))
        return layer
    
    def _make_drm_layer(self, resnext_layer, out_channels, channels, num_layers, stride, stage_index):
        layer = nn.HybridSequential(prefix='stage%d_drm_'%stage_index)
        with layer.name_scope():
            for i in range(num_layers-1):
                layer.add(TopdownBlock(block_unit=resnext_layer[-i-1], out_channels=channels * 4, channels=channels, cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, stride=1, prefix='block%d_'%(num_layers - i)))
            
            layer.add(TopdownBlock(block_unit=resnext_layer[0], out_channels=out_channels, channels=channels, cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, stride=stride, prefix='block1_'))
        return layer
    
    def topdown(self, F, net, mu, ahat, that, meanfw, varfw, meanfwds, varfwds, bfw, lnpicinput, lnpicpn):
        block_indx = 0
        shape_mu = mu.shape
        loss_mm = F.zeros((shape_mu[0]-1,), ctx=mu.context)
        rpn = F.abs(lnpicinput - lnpicpn)
        
        for layer in net:
            if layer.name.find('stage') != -1:
                for sublayer in layer:
                    mu, loss_mm_val, rpn_val = sublayer(mu, (ahat[block_indx])[::-1], (meanfw[block_indx])[::-1], (varfw[block_indx])[::-1], (meanfwds[block_indx])[::-1], (varfwds[block_indx])[::-1], (bfw[block_indx])[::-1])
                    loss_mm = loss_mm + loss_mm_val
                    rpn = rpn + rpn_val
                    block_indx += 1
                    
            elif layer.name.find('lastblock') != -1:
                mu, loss_mm_val, rpn_val = layer(mu, (ahat[block_indx])[::-1], that, (meanfw[block_indx])[::-1], (varfw[block_indx])[::-1], (bfw[block_indx])[::-1])
                loss_mm = loss_mm + loss_mm_val
                rpn = rpn + rpn_val
                block_indx += 1
            else:
                mu = layer(mu)
        
        return mu, loss_mm, rpn

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
        shape_x = x.shape
        return F.reshape(x, (shape_x[0], self._shape[0], self._shape[1], self._shape[2]))
    
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