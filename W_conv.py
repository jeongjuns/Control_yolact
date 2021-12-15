
import math
import warnings

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.autograd import Variable
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple

#from torch.nn.modules.conv import _ConvNd

flcnt1=0
flcnt2=0
flcnt3=0
avgcnt1=0
avgcnt2=0
avgcnt3=0
#fpnlatlayercnt=0
flfpnlatlayercnt=0
bboxcnt=0
flbboxcnt=0
confcnt=0
flconfcnt=0
maskcnt=0
flmaskcnt=0
makenetcnt=0
flmakenetcnt=0
segcnt=0
flsegcnt=0
# torch.nn.conv2d 변형
class W_ConvNd(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    _in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_1_t,
                 stride: _size_1_t,
                 padding: _size_1_t,
                 dilation: _size_1_t,
                 transposed: bool,
                 output_padding: _size_1_t,
                 groups: int,
                 bias: Optional[Tensor],
                 padding_mode: str) -> None:
        super(W_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))

        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size)) 

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'
            
class W_Conv2d1(W_ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        
        super(W_Conv2d1, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
            
        ################################################# jj add 

        self.W1 = Parameter(make_mw(out_channels, in_channels, kernel_size[0]), requires_grad=True)
        W_Conv2d1.fl = {}
        W_Conv2d1.Wweight={}
        
        ################################################# jj end
        
        
    def _conv_forward(self, input, weight):
        
        if self.padding_mode != 'zeros':
            
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
    def forward(self, input: Tensor) -> Tensor:
                
################################################# jj add 
        
        global flcnt1
        global avgcnt1
        
        if avgcnt1 == 34:
            avgcnt1 = 1
            
        if flcnt1 == 33:
            avgcnt1 += 1
            
        for i in range(33,66):        
            if flcnt1 == i:
                W_Conv2d1.fl['{0}'.format(i-33)] = self.weight.clone().detach()

        if flcnt1 > 32:
            for i in range(1,34):
                if avgcnt1 == i:
                    W_Conv2d1.Wweight['{0}'.format(i)] = mod_compute(W_Conv2d1.fl['{0}'.format(i-1)], self.W1)
                    

        if flcnt1 < 66:
            flcnt1+=1
            
        if 0 < avgcnt1 < 34:
            avgcnt1+=1
            
        if flcnt1 < 34:
            return self._conv_forward(input, self.weight)
        else :
            return self._conv_forward(input, mod_compute(W_Conv2d1.fl['{0}'.format(avgcnt1-2)], self.W1))
        
def mod_compute(fl, w):

    # seungil modification
    if fl.size(3) == 1:
        fl = fl.squeeze(-1).squeeze(-1)
        fla_tensor = w@fl
        fla_tensor = fla_tensor.unsqueeze(-1).unsqueeze(-1)

    elif fl.size(3) == 3:
        fla_tensor = torch.zeros(fl.size(0), fl.size(1), 3, 3)

        for i in range(3):
            for j in range(3):
                temp = fl[:,:,i,j].squeeze(-1).squeeze(-1)
                temp = w@temp
                fla_tensor[:,:,i,j] = temp
                 
    return fla_tensor

def make_mw(o_size, i_size, k_size):

    # seungil modification
    mw = torch.eye(o_size)

    return mw
    
################################################# jj end
            
class W_Conv2d2(W_ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        
        super(W_Conv2d2, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        ################################################# jj add
        
        self.W2 = Parameter(make_mw(out_channels, in_channels, kernel_size[0]), requires_grad=True)
        W_Conv2d2.fl = {}
        W_Conv2d2.Wweight={}
        
        ################################################# jj end
                
    def _conv_forward(self, input, weight):
        
        if self.padding_mode != 'zeros':
            
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
    def forward(self, input: Tensor) -> Tensor:
################################################# jj add
        global flcnt2
        global avgcnt2

        if avgcnt2 == 34:
            avgcnt2 = 1
            
        if flcnt2 == 33:
            avgcnt2 += 1
            
        for i in range(33,66):        
            if flcnt2 == i:
                W_Conv2d2.fl['{0}'.format(i-33)] = self.weight.clone().detach()

        if flcnt2 > 32:
            for i in range(1,34):
                if avgcnt2 == i:
                    W_Conv2d2.Wweight['{0}'.format(i)] = mod_compute(W_Conv2d2.fl['{0}'.format(i-1)], self.W2)

        if flcnt2 < 66:
            flcnt2+=1
            
        if 0 < avgcnt2 < 34:
            avgcnt2+=1
            
        #if flcnt2 == 66:
        #    print(W_Conv2d2.fl['{0}'.format(32)][0][0])
        
        if flcnt2 < 34:
            return self._conv_forward(input, self.weight)
        else :
            return self._conv_forward(input, mod_compute(W_Conv2d2.fl['{0}'.format(avgcnt2-2)], self.W2))

################################################# jj end


class W_Conv2d3(W_ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        
        super(W_Conv2d3, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
            
################################################# jj add    

        self.W3 = Parameter(make_mw(out_channels, in_channels, kernel_size[0]), requires_grad=True)
        W_Conv2d3.fl = {}
        W_Conv2d3.Wweight={}

################################################# jj end
    def _conv_forward(self, input, weight):
        
        if self.padding_mode != 'zeros':
            
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
    def forward(self, input: Tensor) -> Tensor:
    ################################################# jj add
        global flcnt3
        global avgcnt3
        
        if avgcnt3 == 34:
            avgcnt3 = 1
            
        if flcnt3 == 33:
            avgcnt3 += 1
            
        for i in range(33,66):        
            if flcnt3 == i:
                W_Conv2d3.fl['{0}'.format(i-33)] = self.weight.clone().detach()
        
        if flcnt3 > 32:
            for i in range(1,34):
                if avgcnt3 == i:
                    W_Conv2d3.Wweight['{0}'.format(i)] = mod_compute(W_Conv2d3.fl['{0}'.format(i-1)], self.W3)

        if flcnt3 < 66:
            flcnt3+=1
            
        if 0 < avgcnt3 < 34:
            avgcnt3+=1
        
        if flcnt3 < 34:
            return self._conv_forward(input, self.weight)
        else :
            return self._conv_forward(input, mod_compute(W_Conv2d3.fl['{0}'.format(avgcnt3-2)], self.W3))
    ################################################# jj end
     
class bbox_Conv2d(W_ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(bbox_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
            
        self.mw = Parameter(make_mw(out_channels,in_channels,kernel_size[0]), requires_grad=True)
        bbox_Conv2d.fl={}
        bbox_Conv2d.Wweight={}
        
    def _conv_forward(self, input, weight):
        
        if self.padding_mode != 'zeros':

            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        global flbboxcnt
        global bboxcnt
        
        if bboxcnt == 6:
            bboxcnt = 1
            
        if flbboxcnt == 5:
            bboxcnt += 1
            
        for i in range(5,10):        
            if flbboxcnt == i:
                bbox_Conv2d.fl['{0}'.format(i-5)] = self.weight.clone().detach()
        
        if flbboxcnt > 4:
            for i in range(1,6):
                if bboxcnt == i:
                    bbox_Conv2d.Wweight['{0}'.format(i)] = mod_compute(bbox_Conv2d.fl['{0}'.format(i-1)], self.mw)

        if flbboxcnt < 10:
            flbboxcnt+=1
            
        if 0 < bboxcnt < 6:
            bboxcnt+=1
        
        #if flbboxcnt == 10:
        #    print(bbox_Conv2d.fl['{0}'.format(0)][0][0])
        #    print(bbox_Conv2d.Wweight['{0}'.format(1)][0][0])

        
        if flbboxcnt < 6:
            return self._conv_forward(input, self.weight)
        else :
            return self._conv_forward(input, mod_compute(bbox_Conv2d.fl['{0}'.format(bboxcnt-2)], self.mw))

class conf_Conv2d(W_ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(conf_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self.mw = Parameter(make_mw(out_channels,in_channels,kernel_size[0]), requires_grad=True)
        conf_Conv2d.fl={}
        conf_Conv2d.Wweight={}
        
    def _conv_forward(self, input, weight):
        
        if self.padding_mode != 'zeros':

            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        global flconfcnt
        global confcnt
        
        if confcnt == 6:
            confcnt = 1
            
        if flconfcnt == 5:
            confcnt += 1
            
        for i in range(5,10):        
            if flconfcnt == i:
                conf_Conv2d.fl['{0}'.format(i-5)] = self.weight.clone().detach()
        
        if flconfcnt > 4:
            for i in range(1,6):
                if confcnt == i:
                    conf_Conv2d.Wweight['{0}'.format(i)] = mod_compute(conf_Conv2d.fl['{0}'.format(i-1)], self.mw)

        if flconfcnt < 10:
            flconfcnt+=1
            
        if 0 < confcnt < 6:
            confcnt+=1
            
            
        if flconfcnt < 6:
            return self._conv_forward(input, self.weight)
        else :
            return self._conv_forward(input, mod_compute(conf_Conv2d.fl['{0}'.format(confcnt-2)], self.mw))
        

class mask_Conv2d(W_ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(mask_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        
        self.mw = Parameter(make_mw(out_channels,in_channels,kernel_size[0]), requires_grad=True)
        mask_Conv2d.fl={}
        mask_Conv2d.Wweight={}
        
    def _conv_forward(self, input, weight):
        
        if self.padding_mode != 'zeros':

            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
    
        global flmaskcnt
        global maskcnt
        
        if maskcnt == 6:
            maskcnt = 1
            
        if flmaskcnt == 5:
            maskcnt += 1
            
        for i in range(5,10):        
            if flmaskcnt == i:
                mask_Conv2d.fl['{0}'.format(i-5)] = self.weight.clone().detach()
        
        if flmaskcnt > 4:
            for i in range(1,6):
                if maskcnt == i:
                    mask_Conv2d.Wweight['{0}'.format(i)] = mod_compute(mask_Conv2d.fl['{0}'.format(i-1)], self.mw)

        if flmaskcnt < 10:
            flmaskcnt+=1
            
        if 0 < maskcnt < 6:
            maskcnt+=1
        
        if flmaskcnt < 6:
            return self._conv_forward(input, self.weight)
        else :
            return self._conv_forward(input, mod_compute(mask_Conv2d.fl['{0}'.format(maskcnt-2)], self.mw))
        
class makenet_Conv2d(W_ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(makenet_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        
        self.mw = Parameter(make_mw(out_channels,in_channels,kernel_size[0]), requires_grad=True)
        makenet_Conv2d.fl={}
        makenet_Conv2d.Wweight={}

    def _conv_forward(self, input, weight):
        
        if self.padding_mode != 'zeros':

            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
    
        global flmakenetcnt
        global makenetcnt
        if makenetcnt == 11:
            makenetcnt = 1
  
        if flmakenetcnt == 10:
            makenetcnt += 1
            
        for i in range(10,20):        
            if flmakenetcnt == i:
                makenet_Conv2d.fl['{0}'.format(i-10)] = self.weight.clone().detach()
        
        if flmakenetcnt > 9:
            for i in range(1,11):
                if makenetcnt == i:
                    makenet_Conv2d.Wweight['{0}'.format(i)] = mod_compute(makenet_Conv2d.fl['{0}'.format(i-1)], self.mw)

        if flmakenetcnt < 20:
            flmakenetcnt+=1
            
        if 0 < makenetcnt < 11:
            makenetcnt+=1
                
        if flmakenetcnt < 11:
            return self._conv_forward(input, self.weight)
        else :
            return self._conv_forward(input, mod_compute(makenet_Conv2d.fl['{0}'.format(makenetcnt-2)], self.mw))
        
class seg_Conv2d(W_ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(seg_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        
        self.mw = Parameter(make_mw(out_channels,in_channels,kernel_size[0]), requires_grad=True)
        seg_Conv2d.fl={}
        seg_Conv2d.Wweight={}
        
    def _conv_forward(self, input, weight):
        
        if self.padding_mode != 'zeros':

            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        global flsegcnt
        global segcnt
        
        if segcnt == 2:
            segcnt = 1
            
        if flsegcnt == 1:
            segcnt += 1
            
        for i in range(1,2):        
            if flsegcnt == i:
                seg_Conv2d.fl['{0}'.format(i-1)] = self.weight.clone().detach()
        
        if flsegcnt > 0:
            for i in range(1,2):
                if segcnt == i:
                    seg_Conv2d.Wweight['{0}'.format(i)] = mod_compute(seg_Conv2d.fl['{0}'.format(i-1)], self.mw)

        if flsegcnt < 2:
            flsegcnt+=1
            
        if 0 < segcnt < 2:
            segcnt+=1
        
        if flsegcnt < 2:
            return self._conv_forward(input, self.weight)
        else :
            return self._conv_forward(input, mod_compute(seg_Conv2d.fl['{0}'.format(segcnt-2)], self.mw))
            
class fpn_lat_layers_Conv2d(W_ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(fpn_lat_layers_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
            
        self.mw = Parameter(make_mw(out_channels,in_channels,kernel_size[0]), requires_grad=True)
        self.fl_2048=torch.ones(256,2048,1,1)
        self.fl_1024=torch.ones(256,1024,1,1)
        self.fl_512=torch.ones(256,512,1,1)
        
        self.fla_2048=torch.ones(256,2048,1,1)
        self.fla_1024=torch.ones(256,1024,1,1)
        self.fla_512=torch.ones(256,512,1,1)
        self.in_channels = in_channels
        self.cnt = 0
        
    def _conv_forward(self, input, weight):
        
        if self.padding_mode != 'zeros':

            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:

        if self.cnt < 2:
            self.cnt += 1

        if self.cnt == 2:
            if self.in_channels == 2048:
                self.fl_2048 = self.weight.clone().detach()

            elif self.in_channels == 1024: 
                self.fl_1024 = self.weight.clone().detach()

            elif self.in_channels == 512:
                self.fl_512 = self.weight.clone().detach()
            self.cnt += 1
        
        if self.cnt > 2:
            if self.in_channels == 2048:
                self.fla_2048 = self.fl_2048.squeeze(-1).squeeze(-1)
                self.fla_2048 = self.mw@self.fla_2048
                self.fla_2048 = self.fla_2048.unsqueeze(-1).unsqueeze(-1)     

            elif self.in_channels == 1024:
                self.fla_1024 = self.fl_1024.squeeze(-1).squeeze(-1)
                self.fla_1024 = self.mw@self.fla_1024
                self.fla_1024 = self.fla_1024.unsqueeze(-1).unsqueeze(-1)     

            elif self.in_channels == 512:
                self.fla_512 = self.fl_512.squeeze(-1).squeeze(-1)
                self.fla_512 = self.mw@self.fla_512
                self.fla_512 = self.fla_512.unsqueeze(-1).unsqueeze(-1)     
        
        #print(self.fl_512[0][0])
        
        if self.cnt < 2:
            return self._conv_forward(input, self.weight)
        elif self.in_channels == 2048:
            return self._conv_forward(input, self.fla_2048)
        elif self.in_channels == 1024:
            return self._conv_forward(input, self.fla_1024)
        elif self.in_channels == 512:
            return self._conv_forward(input, self.fla_512)

        return self._conv_forward(input, self.weight)

class fpn_pred_layers_Conv2d(W_ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        x_cnt : int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(fpn_pred_layers_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
            
        self.mw = Parameter(make_mw(out_channels,in_channels,kernel_size[0]), requires_grad=True)
        self.fl_512=torch.ones(256,256,3,3)
        self.fl_1024=torch.ones(256,256,3,3)
        self.fl_2048=torch.ones(256,256,3,3)

        self.fla_512=torch.ones(256,256,3,3)
        self.fla_1024=torch.ones(256,256,3,3)
        self.fla_2048=torch.ones(256,256,3,3)

        self.cnt = 0
        self.x_cnt = x_cnt
        
    def _conv_forward(self, input, weight):
        
        if self.padding_mode != 'zeros':

            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:

        if self.cnt < 2:
            self.cnt += 1

        if self.cnt == 2:
            if self.x_cnt == 512:
                self.fl_512 = self.weight.clone().detach()

            elif self.x_cnt == 1024: 
                self.fl_1024 = self.weight.clone().detach()

            elif self.x_cnt == 2048:
                self.fl_2048 = self.weight.clone().detach()
            self.cnt += 1

        if self.cnt > 2:
            if self.x_cnt == 512:
                self.fla_512 = torch.zeros(self.fl_512.size(0), self.fl_512.size(1),3,3).cuda()
                for i in range(3):
                    for j in range(3):
                        temp = self.fl_512[:,:,i,j].squeeze(-1).squeeze(-1)
                        temp = self.mw@temp
                        self.fla_512[:,:,i,j] = temp   

            if self.x_cnt == 1024:
                self.fla_1024 = torch.zeros(self.fl_1024.size(0), self.fl_1024.size(1),3,3).cuda()
                for i in range(3):
                    for j in range(3):
                        temp = self.fl_1024[:,:,i,j].squeeze(-1).squeeze(-1)
                        temp = self.mw@temp
                        self.fla_1024[:,:,i,j] = temp  
                        
            if self.x_cnt == 2048:
                self.fla_2048 = torch.zeros(self.fl_2048.size(0), self.fl_2048.size(1),3,3).cuda()
                for i in range(3):
                    for j in range(3):
                        temp = self.fl_2048[:,:,i,j].squeeze(-1).squeeze(-1)
                        temp = self.mw@temp
                        self.fla_2048[:,:,i,j] = temp    

        
        if self.cnt < 2:
            return self._conv_forward(input, self.weight)
        elif self.x_cnt == 512:
            return self._conv_forward(input, self.fla_512)
        elif self.x_cnt == 1024:
            return self._conv_forward(input, self.fla_1024)
        elif self.x_cnt == 2048:
            return self._conv_forward(input, self.fla_2048)
        return self._conv_forward(input, self.weight)

class fpn_down_layers_Conv2d(W_ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        x_cnt : int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(fpn_down_layers_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
            
        self.mw = Parameter(make_mw(out_channels,in_channels,kernel_size[0]), requires_grad=True)
        self.fl_0=torch.ones(2,2,2,2)
        self.fl_1=torch.ones(2,2,2,2)

        self.fla_0=torch.ones(2,2,2,2)
        self.fla_1=torch.ones(2,2,2,2)

        self.cnt = 0
        self.x_cnt = x_cnt
        
    def _conv_forward(self, input, weight):
        
        if self.padding_mode != 'zeros':

            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:

        if self.cnt < 2:
            self.cnt += 1

        if self.cnt == 2:
            if self.x_cnt == 0:
                self.fl_0 = self.weight.clone().detach()

            elif self.x_cnt == 1: 
                self.fl_1 = self.weight.clone().detach()

            self.cnt += 1

        if self.cnt > 2:
            if self.x_cnt == 0:
                self.fla_0 = torch.zeros(self.fl_0.size(0), self.fl_0.size(1),3,3).cuda()
                for i in range(3):
                    for j in range(3):
                        temp = self.fl_0[:,:,i,j].squeeze(-1).squeeze(-1)
                        temp = self.mw@temp
                        self.fla_0[:,:,i,j] = temp   

            if self.x_cnt == 1:
                self.fla_1 = torch.zeros(self.fl_1.size(0), self.fl_1.size(1),3,3).cuda()
                for i in range(3):
                    for j in range(3):
                        temp = self.fl_1[:,:,i,j].squeeze(-1).squeeze(-1)
                        temp = self.mw@temp
                        self.fla_1[:,:,i,j] = temp  
                   
        
        if self.cnt < 2:
            return self._conv_forward(input, self.weight)
        elif self.x_cnt == 0:
            return self._conv_forward(input, self.fla_0)
        elif self.x_cnt == 1:
            return self._conv_forward(input, self.fla_1)

        return self._conv_forward(input, self.weight)