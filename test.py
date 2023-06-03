import torch

if_cuda = torch.cuda.is_available()
print("if_cuda=",if_cuda)
gpu_count = torch.cuda.device_count()
print("gpu_count=",torch.cuda.current_device())
