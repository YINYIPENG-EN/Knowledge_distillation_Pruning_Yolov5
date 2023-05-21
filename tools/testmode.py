import torch
import torch.nn as nn
PRUNABLE_MODULES = [ nn.modules.conv._ConvNd, nn.modules.batchnorm._BatchNorm, nn.Linear, nn.PReLU]
grad_fn_to_module = {}  # 如果获取不到是无法剪枝的
visited = {}  # visited会记录每层出现的次数
def _record_module_grad_fn(module, inputs, outputs): # 记录model的grad_fn
    if module not in visited:
        visited[module] = 1
    else:
        visited[module] += 1
    grad_fn_to_module[outputs.grad_fn] = module
model = torch.load('../runs/train/exp/weights/best.pt')['model'].float().cpu()
# for para in model.parameters():
#     para.requires_grad = True
x = torch.ones(1, 3, 640, 640)
for m in model.modules():
    if isinstance(m, tuple(PRUNABLE_MODULES)):
        hooks = [m.register_forward_hook(_record_module_grad_fn)]
out = model(x)
for hook in hooks:
    hook.remove()
print(grad_fn_to_module)

