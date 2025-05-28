import torch
print(torch.version.cuda)        # CUDA version PyTorch was built with
print(torch.cuda.is_available()) # True if CUDA works
print(torch.cuda.get_device_name(0))  # Your GPU name
print(torch.cuda.get_device_capability(0))