
from models.yolo import Detect
from models.common import *
from models.experimental import *

def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors 3
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings 返回形式为models.common.Conv
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings  转int
            except NameError:
                pass
        # n是卷积重复次数，即相同层有几个，gd是网络深度，可以控制重复几次，round是可以进行四舍五入
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, SE_Block]:
            c1, c2 = ch[f], args[0]  # ch[f] 表示上一层的输出通道，args[0]为64，128，128，256，256....
            if c2 != no:  # if not output 意思就是只要不等于output的时候就可以控制卷积通道
                # gw是可以控制卷积核个数，控制通道
                c2 = make_divisible(c2 * gw, 8)  # c2 = c2*gw
            # args = [128]->[64,64]-->输入通道为64，输出64,red=16
            args = [c1, c2, *args[1:]]  # 获得新的具体的通道数 第一次循环：3 32 6 2 2
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])

        elif m is Detect:
            args.append([ch[x] for x in f])  # f = [17, 20, 23] args=[nc, anchors,ch[17],ch[20],ch[23]]]
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module  将参数传入模型
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n_, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

cfg = "../models/yolov5s.yaml"
from models.yolo import Model
model = Model(cfg, ch=3, nc=80)
print(model.model[3].conv)
