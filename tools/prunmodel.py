import torch_pruning as tp
from loguru import logger
from models.common import *
from models.experimental import Ensemble
from utils.torch_utils import select_device
"""
剪枝的时候根据模型结构去剪，不要盲目的猜
剪枝完需要进行一个微调训练
"""
def attempt_load(weights, map_location=None, inplace=True):
    from models.yolo import Detect, Model

    model = Ensemble()
    ckpt = torch.load(weights, map_location=map_location)  # load
    model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())  # without layer fuse  权值的加载

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
            if type(m) is Detect:
                if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1], ckpt  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model, ckpt  # return ensemble
@logger.catch
def layer_pruning(weights):
    logger.add('../logs/layer_pruning.log', rotation='1 MB')
    device = select_device('cpu')
    model, ckpt = attempt_load(weights, map_location=device)
    for para in model.parameters():
        para.requires_grad = True
    x = torch.zeros(1, 3, 640, 640)
    # -----------------对整个模型的剪枝--------------------
    strategy = tp.strategy.L1Strategy()
    DG = tp.DependencyGraph()
    DG = DG.build_dependency(model, example_inputs=x)

    """
    这里写要剪枝的层
    这里以backbone为例
    """
    # included_layers = [model.model[3].conv] # 仅仅想剪一个卷积层，
    included_layers = []
    for layer in model.model[:10]:
        if type(layer) is Conv:
            included_layers.append(layer.conv)
        elif type(layer) is C3:
            included_layers.append(layer.cv1.conv)
            included_layers.append(layer.cv2.conv)
            included_layers.append(layer.cv3.conv)
        elif type(layer) is SPPF:
            included_layers.append(layer.cv1.conv)
            included_layers.append(layer.cv2.conv)

    num_params_before_pruning = tp.utils.count_params(model)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m in included_layers:
            pruning_plan = DG.get_pruning_plan(m, tp.prune_conv, idxs=strategy(m.weight, amount=0.4))
            logger.info(pruning_plan)
            # 执行剪枝
            pruning_plan.exec()
    # 获得剪枝以后的参数量
    num_params_after_pruning = tp.utils.count_params(model)
    # 输出一下剪枝前后的参数量
    logger.info("  Params: %s => %s\n" % (num_params_before_pruning, num_params_after_pruning))
    # 剪枝完以后模型的保存(不要用torch.save(model.state_dict(),...))

    model_ = {
        'epoch': ckpt['epoch'],
        'best_fitness': ckpt['best_fitness'],
        'model': model.half(),
        'ema': ckpt['ema'],
        'updates': ckpt['updates'],
        'optimizer': ckpt['optimizer'],
        'wandb_id': ckpt['wandb_id']
    }
    torch.save(model_, '../model_data/layer_pruning.pt')
    del model_, ckpt
    logger.info("剪枝完成\n")


layer_pruning('../runs/train/exp/weights/best.pt')

