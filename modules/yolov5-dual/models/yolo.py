# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization, feature_visualization2
from utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device, time_sync
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

#对特征图检测
class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch2=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes 类别数量coco20为例
        self.no = nc + 5  # number of outputs per anchor 四个坐标信息+目标得分
        self.nl = len(anchors)  # number of detection layers 不同尺度特征图层数
        self.na = len(anchors[0]) // 2  # number of anchors 每个特征图anchors数
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch2)  # output conv x是通道取值 na是3 no是25
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        #logits_ = []  # 修改---1
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                #logits = x[i][..., 5:]  # 修改---2
                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
                #logits_.append(logits.view(bs, -1, self.no - 5))  # 修改---3
                #(torch.cat(z, 1), torch.cat(logits_, 1), x)
        return x if self.training else (torch.cat(z, 1), x) #返回预测框坐标、得分和分类

#划分单元网格
    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

#网络模型
class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3,ch2=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml 获得yaml文件
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict 加载yaml文件 以字典的形式

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        ch2 = self.yaml['ch2'] = self.yaml.get('ch2', ch2)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save, self.backbone1depth= parse_model(deepcopy(self.yaml), ch=[ch],ch2=[ch2])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # 读取Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            x=self.forward(torch.zeros(1, ch, s, s),torch.zeros(1, ch2, s, s))
            
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s),torch.zeros(1, ch2, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m) #检查anchor顺序和stride顺序是否一致
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases 初始化
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    #在测试时候做数据增强
    #python detect.py --weights yolov5s.pt --img 832 --source ./inference/images/ --augment
    #--img大小需要大于640设置为832
    def forward(self, x1,x2, augment=False, profile=False, visualize=False):
        return self._forward_once(x1,x2, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x1,x2):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max())) #图像尺寸改变
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    #输入经过网络每一层
    def _forward_once(self, x1,x2, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.i<self.backbone1depth:
                x1 = m(x1)
                x2=x2
                y.append(x1 if m.i in self.save else None)  # save output
            else:
                if m.f != -1:  # if not from previous layer
                    if m.type == 'models.common.Concat3':
                        for j in m.f:
                            if j==-1:
                                x2 = x2
                            else:
                                x1 = y[j]
                    else:
                        x2 = y[m.f] if isinstance(m.f, int) else [x2 if j == -1 else y[j] for j in m.f]  # from earlier layers

                if profile:
                    self._profile_one_layer(m, x2, dt)
                if m.type == 'models.common.Concat3':
                    x2 = m(x2,x1)  # run
                else:
                    x2 = m(x2)  # run
                    x1=x1
                y.append(x2 if m.i in self.save else None)  # save output
            
            if visualize:
                feature_visualization(x2, m.type, m.i, save_dir=visualize)
            
            feature_vis = False
            if m.type == 'models.common.CBAM' and feature_vis and m.i==7:
                print(m.type, m.i)
                feature_visualization2(x2, m.type, m.i,128,8,16)
        return x2

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    #初始化detect组件的偏置
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    #卷积和归一化进行融合
    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

#解析网络配置文件构建模型
def parse_model(d, ch,ch2):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")#记录日志
    #1、以下是读取配置dict里的参数
    #————————————————————————————————————————————————————————————————
    anchors, nc, gd, gw ,backbone1depth= d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'],d['backbone1depth']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5) 输出的通道数
 
    # 2、开始搭建网络
    #————————————————————————————————————————————————————————————————
    '''
    layers: 保存每一层的层结构
    # save: 记录下所有层结构中from中不是-1的层结构序号
    # c2: 保存当前层的输出channel
    '''
    layers, save, c2,c22 = [], [], ch[-1] ,ch2[-1] # layers, savelist, ch out初始化
    # from(当前层输入来自哪些层), number(当前层次数 初定模型深度), module(当前层类别), args(当前层类参数 初定)
    for i, (f, n, m, args) in enumerate(d['backbone1']+d['backbone2']+d['head']):  # 遍历backbone和head的每一层
        m = eval(m) if isinstance(m, str) else m  #得到当前层的真实类名 
        for j, a in enumerate(args): #循环模块参数args
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass
                
        n = n_ = max(round(n * gd), 1) if n > 1 else n
        
        if i<backbone1depth:
            if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, conv_bn_relu_maxpool, Shuffle_Block,CARAFE,Concat3]:
                c1, c2 = ch[f], args[0]
                if c2 != no:  
                    c2 = make_divisible(c2 * gw, 8)  #保证通道是8的倍数
                # 在初始arg的基础上更新 加入当前层的输入channel并更新当前层
                # [in_channel, out_channel, *args[1:]]xin
                args = [c1, c2, *args[1:]] 
        # depth gain 控制深度  如v5s: n*0.33   n: 当前模块的次数(间接控制深度)
        else : 
            if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                     BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, conv_bn_relu_maxpool, Shuffle_Block,CBAM]:
                c12, c22 = ch2[f], args[0] #保存输出
                if c22 != no:  
                    c22 = make_divisible(c22 * gw, 8)  #保证通道是8的倍数
                args = [c12, c22, *args[1:]] 
                if m in [BottleneckCSP, C3, C3TR, C3Ghost,CBAM]:
                    args.insert(2, n)   #在第二个位置插入bottleneck个数n
                    n = 1  #重置
            elif m is nn.BatchNorm2d: # BN层只需要返回上一层的输出channel
                args = [ch2[f]]
            elif m is Concat2: #Concat：f是所有需要拼接层的索引，则输出通道c2是所有层的和
                c22 = 0
                for x in f:
                    if x==-1:
                        c2p = ch2[x]
                    else:
                        c2p = ch[x]
                    c22 = c22+c2p
            elif m is Concat3: #Concat：f是所有需要拼接层的索引，则输出通道c2是所有层的和
                c22 = 0
                for x in f:
                    if x==-1:
                        c2p = ch2[x]
                    else:
                        c2p = ch[x]
                    c22 = c22+c2p
            elif m is Concat: #Concat：f是所有需要拼接层的索引，则输出通道c2是所有层的和
                c22 = sum(ch2[x] for x in f)
            elif m is Detect:#args先填入每个预测层的输入通道数，然后填入生成所有预测层对应的预测框的初始高宽的列表。
                args.append([ch2[x] for x in f]) #在args中加入三个Detect层的输出channel
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
            elif m is Contract:
                c22 = ch2[f] * args[0] ** 2
            elif m is Expand:
                c22 = ch2[f] // args[0] ** 2
            elif m is MobileOne:
                c12, c22 = ch2[f], args[0]
                c22 = make_divisible(c22 * gw, 8)
                args = [c12, c22, n, *args[1:]]
            else:
                c22 = ch2[f]
        #拿args里的参数去构建了module m，然后模块的循环次数用参数n控制。
        # m_: 得到当前层module  如果n>1就创建多个m(当前层结构), 如果n=1就创建一个m
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # 打印当前层结构的一些基本信息
        np = sum(x.numel() for x in m_.parameters())  # 计算这一层的参数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # 打印日志文件信息（每一层module构建的编号、参数量等）
        # append to savelist  把所有层结构中from不是-1的值记下 
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_) # 将当前层结构module加入layers中
        if i == 0:
            ch = []
        if i<backbone1depth:
            ch.append(c2)
        
        if i == backbone1depth:
            ch2 = ch #层编号是包含backbone1的
        ch2.append(c22)

    return nn.Sequential(*layers), sorted(save),backbone1depth      #当循环结束后再构建成模型


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Test all models
    if opt.test:
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
