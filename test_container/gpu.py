# Test GPU can be seen
# Will Fail on CPU ONLY, Should Pass on NVIDIA

import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))