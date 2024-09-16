import torch

# 检查 CUDA 是否可用
print("CUDA可用:", torch.cuda.is_available())
print("GPU数量:", torch.cuda.device_count())
print("当前GPU设备:", torch.cuda.current_device())
print("当前GPU名称:", torch.cuda.get_device_name(torch.cuda.current_device()))