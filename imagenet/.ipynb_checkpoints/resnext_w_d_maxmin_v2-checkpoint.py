'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py 
Original author Wei Wu

Implemented the following paper:
Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He. "Aggregated Residual Transformations for Deep Neural Network" CVPR 2017 https://arxiv.org/pdf/1611.05431v2.pdf
Jie Hu, Li Shen, Gang Sun. "Squeeze-and-Excitation Networks" https://arxiv.org/pdf/1709.01507v1.pdf

This modification version is based on ResNet v1
This modificaiton version adds dropout layer followed by last pooling layer.
Modified by Lin Xiong Feb-11, 2017
Updated by Lin Xiong Jul-21, 2017
Added Squeeze-and-Excitation block by Lin Xiong Sep-13, 2017
'''
import mxnet as mx

from mxnet.gluon import nn
from mxnet import nd

import numpy as np

def compute_kl_gaussian(F, x, meanfw, varfw, meanvar_indx):
    out = 0.5*F.mean(((F.mean(x, axis=1, exclude=True) - meanfw[meanvar_indx])**2)/(varfw[meanvar_indx] + 1e-8)) + 0.5*F.mean(F.mean((x - F.broadcast_like(F.mean(x, axis=1, exclude=True, keepdims=True), x, lhs_axes=(0,2,3)))**2, axis=1, exclude=True)/(varfw[meanvar_indx] + 1e-8)) - 0.5*F.mean(F.log(F.mean((x - F.broadcast_like(F.mean(x, axis=1, exclude=True, keepdims=True),x,lhs_axes=(0,2,3)))**2, axis=1, exclude=True)+ + 1e-8) - F.log(varfw[meanvar_indx] + 1e-8)) - 0.5
    del meanfw[meanvar_indx]
    del varfw[meanvar_indx]
    return out
    
class BiasAdder(nn.HybridBlock):
    """
    Add a bias into the input
    """
    def __init__ (self, channels, **kwargs):
        super(BiasAdder, self).__init__(**kwargs)
        with self.name_scope():
            self.bias = self.params.get('bias', shape=(1,channels,1,1), allow_deferred_init=True)

    def hybrid_forward(self, F, x, bias):
        activation = x + F.broadcast_like(bias, x, lhs_axes=(0,2,3))

        return activation
    
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
        return F.reshape(x, (-1, self._shape[0], self._shape[1], self._shape[2]))
    
class NReLu(nn.HybridBlock):
    """
    -max(-x,0)
    Parameters
    ----------
    Input shape: (N, C, W, H)
    Output shape: (N, C * W * H)
    """
    def __init__(self, **kwargs):
        super(NReLu, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return -F.Activation(-x, act_type='relu')
    
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

class residual_unit(nn.HybridBlock):
    """Return ResNext Unit symbol for building ResNext
    Parameters
    ----------gl
    data : str
        Input data
    num_filter : int
        Number of output channels
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    bottle_neck : Boolen
        Whether or not to adopt bottle_neck trick as did in ResNet
    num_group : int
        Number of convolution groupes
    bn_mom : float
        Momentum of batch normalization
    workspace : int
        Workspace used in convolution operator
    """
    def __init__(self, in_channels, num_filter, ratio, strides, dim_match, name, num_group,  bn_mom=0.9, **kwargs):
        super(residual_unit, self).__init__(**kwargs)
        self.dim_match = dim_match
        self.num_filter = num_filter
        
        self.in_channels = in_channels
        self.ratio = ratio
        self.strides = strides
        self.num_group = num_group
        self.bn_mom = bn_mom
        
        # block 1
        self.conv1 = nn.Conv2D(in_channels=in_channels, channels=int(num_filter*0.5), kernel_size=(1,1), strides=(1,1), padding=(0,0), use_bias=False, prefix=name + '_conv1_')
        self.bn1 = nn.BatchNorm(in_channels=int(num_filter*0.5), epsilon=2e-5, momentum=bn_mom, prefix=name + '_batchnorm1_')
        self.bn1min = nn.BatchNorm(in_channels=int(num_filter*0.5), epsilon=2e-5, momentum=bn_mom, prefix=name + '_batchnorm1min_')
        self.bias1 = BiasAdder(channels=int(num_filter*0.5), prefix=name + '_bias1_')
        self.relu1 = nn.Activation(activation='relu', prefix=name + '_relu1_')
        self.relu1min = NReLu(prefix=name + '_relu1min_')
        
        # block 2
        self.conv2 = nn.Conv2D(in_channels=int(num_filter*0.5), channels=int(num_filter*0.5), groups=num_group, kernel_size=(3,3), strides=strides, padding=(1,1), use_bias=False, prefix=name + '_conv2_')
        self.bn2 = nn.BatchNorm(in_channels=int(num_filter*0.5), epsilon=2e-5, momentum=bn_mom, prefix=name + '_batchnorm2_')
        self.bn2min = nn.BatchNorm(in_channels=int(num_filter*0.5), epsilon=2e-5, momentum=bn_mom, prefix=name + '_batchnorm2min_')
        self.bias2 = BiasAdder(channels=int(num_filter*0.5), prefix=name + '_bias2_')
        self.relu2 = nn.Activation(activation='relu', prefix=name + '_relu2_')
        self.relu2min = NReLu(prefix=name + '_relu2min_')
        
        # block 3
        self.conv3 = nn.Conv2D(in_channels=int(num_filter*0.5), channels=num_filter, kernel_size=(1,1), strides=(1,1), padding=(0,0), use_bias=False, prefix=name + '_conv3_')
        self.bn3 = nn.BatchNorm(in_channels=num_filter, epsilon=2e-5, momentum=bn_mom, prefix=name + '_batchnorm3_')
        self.bn3min = nn.BatchNorm(in_channels=num_filter, epsilon=2e-5, momentum=bn_mom, prefix=name + '_batchnorm3min_')
        self.bias3 = BiasAdder(channels=num_filter, prefix=name + '_bias3_')
        
        if not dim_match:
            self.fc_sc = nn.Conv2D(in_channels=in_channels, channels=num_filter, kernel_size=(1,1), strides=strides, use_bias=False, prefix=name + '_sc_dense_')
            self.bn_sc = nn.BatchNorm(in_channels=num_filter, epsilon=2e-5, momentum=bn_mom, prefix=name + '_sc_batchnorm_')
            self.bn_scmin = nn.BatchNorm(in_channels=num_filter, epsilon=2e-5, momentum=bn_mom, prefix=name + '_sc_batchnormmin_')
            self.bias_sc = BiasAdder(channels=num_filter, prefix=name + '_bias_sc_')
            
        self.relu3 = nn.Activation(activation='relu', prefix=name + '_relu3_')
        self.relu3min = NReLu(prefix=name + '_relu3min_')
        
    def hybrid_forward(self, F, xmax, xmin):
        shortcutmax = xmax
        shortcutmin = xmin
        
        ahatmax = []; meanfwmax = []; varfwmax = []
        ahatmin = []; meanfwmin = []; varfwmin = []
        
        bfw = []
        
        # block 1
        xmax = self.conv1(xmax)
        meanfwmax.append(F.mean(xmax, axis=1, exclude=True))
        varfwmax.append(F.mean((xmax - F.broadcast_like(F.mean(xmax, axis=1, exclude=True, keepdims=True),xmax, lhs_axes=(0,2,3)))**2, axis=1, exclude=True))

        xmax = self.bn1(xmax)
        xmax = self.bias1(xmax)
        bfw.append(self.bias1.bias)
        xmax = self.relu1(xmax)
        ahatmax.append(xmax.__gt__(0))
        
        xmin = self.conv1(xmin)
        meanfwmin.append(F.mean(xmin, axis=1, exclude=True))
        varfwmin.append(F.mean((xmin - F.broadcast_like(F.mean(xmin, axis=1, exclude=True, keepdims=True),xmin, lhs_axes=(0,2,3)))**2, axis=1, exclude=True))
        
        xmin = self.bn1min(xmin)
        xmin = self.bias1(xmin)
        xmin = self.relu1min(xmin)
        ahatmin.append(xmin.__lt__(0))
        
        # block 2
        xmax = self.conv2(xmax)
        meanfwmax.append(F.mean(xmax, axis=1, exclude=True))
        varfwmax.append(F.mean((xmax - F.broadcast_like(F.mean(xmax, axis=1, exclude=True, keepdims=True),xmax, lhs_axes=(0,2,3)))**2, axis=1, exclude=True))
        
        xmax = self.bn2(xmax)
        xmax = self.bias2(xmax)
        bfw.append(self.bias2.bias)
        xmax = self.relu2(xmax)
        ahatmax.append(xmax.__gt__(0))
        
        xmin = self.conv2(xmin)
        meanfwmin.append(F.mean(xmin, axis=1, exclude=True))
        varfwmin.append(F.mean((xmin - F.broadcast_like(F.mean(xmin, axis=1, exclude=True, keepdims=True),xmin, lhs_axes=(0,2,3)))**2, axis=1, exclude=True))
        
        xmin = self.bn2min(xmin) 
        xmin = self.bias2(xmin)
        xmin = self.relu2min(xmin)
        ahatmin.append(xmin.__lt__(0))
        
        # block 3
        xmax = self.conv3(xmax)
        meanfwmax.append(F.mean(xmax, axis=1, exclude=True))
        varfwmax.append(F.mean((xmax - F.broadcast_like(F.mean(xmax, axis=1, exclude=True, keepdims=True), xmax, lhs_axes=(0,2,3)))**2, axis=1, exclude=True))
        
        xmax = self.bn3(xmax)
        xmax = self.bias3(xmax)
        bfw.append(self.bias3.bias)
        
        xmin = self.conv3(xmin)
        meanfwmin.append(F.mean(xmin, axis=1, exclude=True))
        varfwmin.append(F.mean((xmin - F.broadcast_like(F.mean(xmin, axis=1, exclude=True, keepdims=True), xmin, lhs_axes=(0,2,3)))**2, axis=1, exclude=True))
        
        xmin = self.bn3min(xmin)
        xmin = self.bias3(xmin)
        
        if not self.dim_match:
            shortcutmax = self.fc_sc(shortcutmax)
            meanfwmax.append(F.mean(shortcutmax, axis=1, exclude=True))
            varfwmax.append(F.mean((shortcutmax - F.broadcast_like(F.mean(shortcutmax, axis=1, exclude=True, keepdims=True),shortcutmax, lhs_axes=(0,2,3)))**2, axis=1, exclude=True))
            
            shortcutmax = self.bn_sc(shortcutmax)
            shortcutmax = self.bias_sc(shortcutmax)
            bfw.append(self.bias_sc.bias)
            
            shortcutmin = self.fc_sc(shortcutmin)
            meanfwmin.append(F.mean(shortcutmin, axis=1, exclude=True))
            varfwmin.append(F.mean((shortcutmin - F.broadcast_like(F.mean(shortcutmin, axis=1, exclude=True, keepdims=True),shortcutmin, lhs_axes=(0,2,3)))**2, axis=1, exclude=True))
            
            shortcutmin = self.bn_scmin(shortcutmin)
            shortcutmin = self.bias_sc(shortcutmin)
        
        xmax = xmax + shortcutmax
        xmin = xmin + shortcutmin
        
        xmax = self.relu3(xmax)
        ahatmax.append(xmax.__gt__(0))
        xmin = self.relu3min(xmin)
        ahatmin.append(xmin.__lt__(0))
        
        ahatmax=ahatmax[::-1]
        ahatmin=ahatmin[::-1]
        meanfwmax=meanfwmax[::-1]
        varfwmax=varfwmax[::-1]
        meanfwmin=meanfwmin[::-1]
        varfwmin=varfwmin[::-1]
        bfw=bfw[::-1]
        
        return xmax, xmin, ahatmax, ahatmin, meanfwmax, varfwmax, meanfwmin, varfwmin, bfw

class topdown_residual_unit(nn.HybridBlock):
    """Return ResNext Unit symbol for building ResNext
    Parameters
    ----------gl
    data : str
        Input data
    num_filter : int
        Number of output channels
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    bottle_neck : Boolen
        Whether or not to adopt bottle_neck trick as did in ResNet
    num_group : int
        Number of convolution groupes
    bn_mom : float
        Momentum of batch normalization
    workspace : int
        Workspace used in convolution operator
    """
    def __init__(self, fwblock, name, **kwargs):
        super(topdown_residual_unit, self).__init__(**kwargs)
        self.dim_match = fwblock.dim_match
        self.num_filter = fwblock.num_filter
        self.out_channels = fwblock.in_channels
        self.ratio = fwblock.ratio
        self.strides = fwblock.strides
        self.num_group = fwblock.num_group
        self.bn_mom = fwblock.bn_mom
        
        output_padding = 1 if self.strides[0] > 1 else 0
        
        if not self.dim_match:
            self.fc_sc = nn.Conv2DTranspose(channels=self.out_channels, in_channels=self.num_filter, kernel_size=(1,1), strides=self.strides, use_bias=False, output_padding=output_padding, params=fwblock.fc_sc.params, prefix=name + '_tdsc_dense_')
            
        # block 3
        self.bn3 = nn.BatchNorm(in_channels=self.num_filter, epsilon=2e-5, momentum=self.bn_mom, prefix=name + '_td_batchnorm3_')
        self.bn3min = nn.BatchNorm(in_channels=self.num_filter, epsilon=2e-5, momentum=self.bn_mom, prefix=name + '_td_batchnorm3min_')
        self.conv3 = nn.Conv2DTranspose(channels=int(self.num_filter*0.5), in_channels=self.num_filter, kernel_size=(1,1), use_bias=False, params=fwblock.conv3.params, prefix=name + '_td_conv3_')
        
        # block 2
        self.bn2 = nn.BatchNorm(in_channels=int(self.num_filter*0.5), epsilon=2e-5, momentum=self.bn_mom, prefix=name + '_td_batchnorm2_')
        self.bn2min = nn.BatchNorm(in_channels=int(self.num_filter*0.5), epsilon=2e-5, momentum=self.bn_mom, prefix=name + '_td_batchnorm2min_')
        self.conv2 = nn.Conv2DTranspose(channels=int(self.num_filter*0.5), in_channels=int(self.num_filter*0.5), kernel_size=(3,3), strides=self.strides, padding=(1,1), output_padding=output_padding, use_bias=False, groups=self.num_group, params=fwblock.conv2.params, prefix=name + '_td_conv2_')
        
        # block 1
        self.bn1 = nn.BatchNorm(in_channels=int(self.num_filter*0.5), epsilon=2e-5, momentum=self.bn_mom, prefix=name + '_td_batchnorm1_')
        self.bn1min = nn.BatchNorm(in_channels=int(self.num_filter*0.5), epsilon=2e-5, momentum=self.bn_mom, prefix=name + '_td_batchnorm1min_')
        self.conv1 = nn.Conv2DTranspose(channels=self.out_channels, in_channels=int(self.num_filter*0.5), kernel_size=(1,1), use_bias=False, params=fwblock.conv1.params, prefix=name + '_td_conv1_')
    
    def hybrid_forward(self, F, xmax, xmin, ahatmax, ahatmin, meanfwmax, varfwmax, meanfwmin, varfwmin, bfw):
        if not self.dim_match:
            loss_mmmax = compute_kl_gaussian(F, xmax, meanfwmax, varfwmax, 0)
            loss_mmmax = loss_mmmax + compute_kl_gaussian(F, xmax, meanfwmax, varfwmax, 0)
        else:
            loss_mmmax = compute_kl_gaussian(F, xmax, meanfwmax, varfwmax, 0)

        xmax = self.bn3(xmax)
        
        if not self.dim_match:
            loss_mmmin = compute_kl_gaussian(F, xmin, meanfwmin, varfwmin, 0) 
            loss_mmmin = loss_mmmin + compute_kl_gaussian(F, xmin, meanfwmin, varfwmin, 0)
        else:
            loss_mmmin = compute_kl_gaussian(F, xmin, meanfwmin, varfwmin, 0)
            
        xmin = self.bn3min(xmin)
        
        xmax = xmax * ahatmax[0]
        xmin = xmin * ahatmin[0]
        del ahatmax[0]
        del ahatmin[0]
        
        if not self.dim_match:
            rpn = F.mean(-F.abs((xmax-xmin)*F.broadcast_like(bfw[0].var(), xmax, lhs_axes=(0,2,3))), axis=0, exclude=True)
            del bfw[0]
            rpn = rpn + F.mean(-F.abs((xmax-xmin)*F.broadcast_like(bfw[0].var(), xmax, lhs_axes=(0,2,3))), axis=0, exclude=True)
            del bfw[0]
        else:
            rpn = F.mean(-F.abs((xmax-xmin)*F.broadcast_like(bfw[0].var(), xmax, lhs_axes=(0,2,3))), axis=0, exclude=True)
            del bfw[0]
        
        shortcutmax = xmax
        shortcutmin = xmin
        
        if not self.dim_match:
            shortcutmax = self.fc_sc(shortcutmax)
            shortcutmin = self.fc_sc(shortcutmin)
            
        # block 3
        xmax = self.conv3(xmax)
        xmin = self.conv3(xmin)
        
        # block 2
        loss_mmmax = loss_mmmax + compute_kl_gaussian(F, xmax, meanfwmax, varfwmax, 0)
        xmax = self.bn2(xmax)
        
        loss_mmmin = loss_mmmin + compute_kl_gaussian(F, xmin, meanfwmin, varfwmin, 0)
        xmin = self.bn2min(xmin)
        
        xmax = xmax * ahatmax[0]
        xmin = xmin * ahatmin[0]
        del ahatmax[0]
        del ahatmin[0]
        
        rpn = rpn + F.mean(-F.abs((xmax-xmin)*F.broadcast_like(bfw[0].var(), xmax, lhs_axes=(0,2,3))), axis=0, exclude=True)
        del bfw[0]
        
        xmax = self.conv2(xmax)
        xmin = self.conv2(xmin)
        
        # block 1
        loss_mmmax = loss_mmmax + compute_kl_gaussian(F, xmax, meanfwmax, varfwmax, 0)
        xmax = self.bn1(xmax)
        
        loss_mmmin = loss_mmmin + compute_kl_gaussian(F, xmin, meanfwmin, varfwmin, 0)
        xmin = self.bn1min(xmin)
        
        xmax = xmax * ahatmax[0]
        xmin = xmin * ahatmin[0]
        del ahatmax[0]
        del ahatmin[0]
        
        rpn = rpn + F.mean(-F.abs((xmax-xmin)*F.broadcast_like(bfw[0].var(), xmax, lhs_axes=(0,2,3))), axis=0, exclude=True)
        del bfw[0]
        
        xmax = self.conv1(xmax)
        xmin = self.conv1(xmin)
        
        # combine
        xmax = xmax + shortcutmax
        xmin = xmin + shortcutmin
        
        return xmax, xmin, loss_mmmax, loss_mmmin, rpn
              
class resnext(nn.HybridBlock):
    """Return ResNeXt symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Ouput size of symbol
    num_groupes: int
		Number of convolution groups
    drop_out : float
        Probability of an element to be zeroed. Default = 0.0
    data_type : str
        Dataset type, only cifar10, imagenet and vggface supports
    workspace : int
        Workspace used in convolution operator
    """
    def __init__(self, units, num_stage, filter_list, ratio_list, num_class, num_group, data_type, drop_out, bn_mom=0.9, **kwargs):
        super(resnext, self).__init__(**kwargs)
        num_unit = len(units)
        assert(num_unit == num_stage)
        self.num_class = num_class
        
        # fw
        self.conv0 = nn.Conv2D(in_channels=3, channels=filter_list[0], kernel_size=(7,7), strides=(2,2), padding=(3,3), use_bias=False, prefix='conv0_')
        self.bn0 = nn.BatchNorm(in_channels=filter_list[0], epsilon=2e-5, momentum=bn_mom, prefix='batchnorm0_')
        self.bias0 = BiasAdder(channels=filter_list[0], prefix='bias0_')
        self.relu0 = nn.Activation(activation='relu', prefix='relu0_')
        self.relu0min = NReLu(prefix='relu0min_')
        self.pool0 = nn.MaxPool2D(pool_size=(2,2), strides=(2,2), padding=(0,0), prefix='pool0_')
        
        # td
        self.upsample0 = UpsampleLayer(size=2, scale=1., prefix='up0_')
        self.bntd0 = nn.BatchNorm(in_channels=filter_list[0], epsilon=2e-5, momentum=bn_mom, prefix='td_batchnorm0_')
        self.bntd0min = nn.BatchNorm(in_channels=filter_list[0], epsilon=2e-5, momentum=bn_mom, prefix='td_batchnorm0min_')
        self.tdconv0 = nn.Conv2DTranspose(channels=3, in_channels=filter_list[0], kernel_size=(7,7), strides=(2,2), padding=(3,3), output_padding=1, use_bias=False, params=self.conv0.params, prefix='td_conv0_')
        
        self.residual_stages = nn.HybridSequential(prefix='residual_')
        topdown_list = []
        for i in range(num_stage):
            self.residual_stages.add(residual_unit(in_channels=filter_list[i], num_filter=filter_list[i+1], ratio=ratio_list[2], strides=(1 if i==0 else 2, 1 if i==0 else 2), dim_match=False, name='stage%d_unit%d' % (i + 1, 1), num_group=num_group, bn_mom=bn_mom, prefix='stage%d_unit%d_' % (i + 1, 1)))
            topdown_list.append(topdown_residual_unit(fwblock=self.residual_stages[-1], name='stage%d_td_unit%d' % (i + 1, 1), prefix='stage%d_td_unit%d_' % (i + 1, 1)))
            for j in range(units[i]-1):
                self.residual_stages.add(residual_unit(in_channels=filter_list[i+1], num_filter=filter_list[i+1], ratio=ratio_list[2], strides=(1,1), dim_match=True, name='stage%d_unit%d' % (i + 1, j + 2), num_group=num_group, bn_mom=bn_mom, prefix='stage%d_unit%d_' % (i + 1, j + 2)))
                topdown_list.append(topdown_residual_unit(fwblock=self.residual_stages[-1], name='stage%d_td_unit%d' % (i + 1, j + 2), prefix='stage%d_td_unit%d_' % (i + 1, j + 2)))
        
        with self.name_scope():
            self.topdown_stages = nn.HybridSequential(prefix='td_residual_')
            for block in topdown_list[::-1]:
                self.topdown_stages.add(block)
        
        # fw classifier
        self.pool1 = nn.GlobalAvgPool2D(prefix='pool1_')
        self.drop1 = nn.Dropout(rate=drop_out, prefix='dp1_')
        self.fc = nn.Conv2D(in_channels=filter_list[-1], channels=num_class, kernel_size=(1, 1), use_bias=True, prefix='dense_')
        self.flatten1 = nn.Flatten(prefix='flatten1_')
        
        # bw classifier
        self.reshape = Reshape(shape=(num_class, 1, 1),prefix='reshape_')
        self.td_drop1 = nn.Dropout(rate=drop_out, prefix='td_dp1_')
        self.td_fc = nn.Conv2DTranspose(channels=filter_list[-1], in_channels=num_class, kernel_size=(1,1), strides=(1, 1), use_bias=False, params=self.fc.params, prefix='td_dense_')
        self.upsample1 = UpsampleLayer(size=7, scale=1./(7**2), prefix='up1_')
        
    def hybrid_forward(self, F, x, y, cond):
        ahatmax = []; meanfwmax = []; varfwmax = []
        ahatmin = []; meanfwmin = []; varfwmin = []
        
        bfw = []
        
        x = F.cast(x, 'float16')
        y = F.cast(y, 'float16')
        cond = F.cast(cond, 'float16')
        
        x = self.conv0(x)
        meanfwmax.append(F.mean(x, axis=1, exclude=True))
        varfwmax.append(F.mean((x - F.broadcast_like(F.mean(x, axis=1, exclude=True, keepdims=True), x, lhs_axes=(0,2,3)))**2, axis=1, exclude=True))
        meanfwmin.append(meanfwmax[-1])
        varfwmin.append(varfwmax[-1])

        x = self.bn0(x)
        x = self.bias0(x)
        bfw.append(self.bias0.bias)
        
        xmax = self.relu0(x)
        ahatmax.append(xmax.__gt__(0))
        
        xmin = self.relu0min(x)
        ahatmin.append(xmin.__lt__(0))
        
        xmax_pool = self.pool0(xmax)
        thatmax = (xmax-F.repeat(F.repeat(xmax_pool, repeats=2, axis=2), repeats=2, axis=3)).__ge__(0)
        
        xmin_pool = -self.pool0(-xmin)
        thatmin = (xmin-F.repeat(F.repeat(xmin_pool, repeats=2, axis=2), repeats=2, axis=3)).__le__(0)
        
        xmax = xmax_pool
        xmin = xmin_pool
        
        for res_unit in self.residual_stages:
            xmax, xmin, ahatmax_res, ahatmin_res, meanfwmax_res, varfwmax_res, meanfwmin_res, varfwmin_res, bfw_res = res_unit(xmax, xmin)
            ahatmax.append(ahatmax_res)
            ahatmin.append(ahatmin_res)
            meanfwmax.append(meanfwmax_res)
            meanfwmin.append(meanfwmin_res)
            varfwmax.append(varfwmax_res)
            varfwmin.append(varfwmin_res)
            bfw.append(bfw_res)
        
        xmax = self.pool1(xmax)
        xmax = self.drop1(xmax)
        xmax = self.fc(xmax)
        xmax = self.flatten1(xmax)
        
        xmin = self.pool1(xmin)
        xmin = self.drop1(xmin)
        xmin = self.fc(xmin)
        xmin = self.flatten1(xmin)
        
        ahatmax = ahatmax[::-1]
        ahatmin = ahatmin[::-1]
        meanfwmax = meanfwmax[::-1]
        meanfwmin = meanfwmin[::-1]
        varfwmax = varfwmax[::-1]
        varfwmin = varfwmin[::-1]
        bfw = bfw[::-1]
        
        x = 0.5*xmax + 0.5*xmin
        c = F.where(cond, y, F.argmax(x, axis=1))
        xmu = F.one_hot(c, self.num_class)
        
        xhatmax, xhatmin, loss_mmmax, loss_mmmin, rpn = self.topdown(F, self.topdown_stages, xmu, ahatmax, ahatmin, thatmax, thatmin, meanfwmax, meanfwmin, varfwmax, varfwmin, bfw)
        
        del ahatmax, ahatmin, thatmax, thatmin, meanfwmax, meanfwmin, varfwmax, varfwmin, bfw
        
        return xmax, xmin, xhatmax, xhatmin, loss_mmmax, loss_mmmin, rpn
    
    def topdown(self, F, net, x, ahatmax, ahatmin, thatmax, thatmin, meanfwmax, meanfwmin, varfwmax, varfwmin, bfw):
        x = F.cast(x, 'float16')
        
        x = self.reshape(x)
        x = self.td_drop1(x)
        x = self.td_fc(x)
        x = self.upsample1(x)
        
        xmax = x
        xmin = x
        
        for i, (td_unit, amax, amin, meanmax, meanmin, varmax, varmin, b) in enumerate(zip(net, ahatmax[0:-1], ahatmin[0:-1], meanfwmax[0:-1], meanfwmin[0:-1], varfwmax[0:-1], varfwmin[0:-1], bfw[0:-1])):
            xmax, xmin, loss_mmmax_res, loss_mmmin_res, rpn_res = td_unit(xmax, xmin, amax, amin, meanmax, meanmin, varmax, varmin, b)
            if i == 0:
                loss_mmmax = loss_mmmax_res
                loss_mmmin = loss_mmmin_res
                rpn = rpn_res
            else:
                loss_mmmax = loss_mmmax + loss_mmmax_res
                loss_mmmin = loss_mmmin + loss_mmmin_res
                rpn = rpn + rpn_res
        
        loss_mmmax = loss_mmmax + compute_kl_gaussian(F, xmax, meanfwmax, varfwmax, -1)
        xmax = self.bntd0(xmax)
        
        loss_mmmin = loss_mmmin + compute_kl_gaussian(F, xmin, meanfwmin, varfwmin, -1)
        xmin = self.bntd0min(xmin)
        
        xmax = self.upsample0(xmax)
        xmax = xmax * thatmax
        xmax = xmax * ahatmax[-1]
        del ahatmax[-1]
        del thatmax
        
        xmin = self.upsample0(xmin)
        xmin = xmin * thatmin
        xmin = xmin * ahatmin[-1]
        del ahatmin[-1]
        del thatmin
        
        rpn = rpn + F.mean(-F.abs((xmax-xmin)*F.broadcast_like(bfw[-1].var(), xmax, lhs_axes=(0,2,3))), axis=0, exclude=True)
        del bfw[-1]
        
        xmax = self.tdconv0(xmax)
        xmin = self.tdconv0(xmin)
        
        return xmax, xmin, loss_mmmax, loss_mmmin, rpn
        
        
        
    