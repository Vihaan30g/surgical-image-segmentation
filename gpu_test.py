import torch

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():

    print("GPU:", torch.cuda.get_device_name(0))

    x = torch.randn(1000, 1000).cuda()

    y = torch.randn(1000, 1000).cuda()

    z = torch.matmul(x, y)

    print("GPU Test Passed")
