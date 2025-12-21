import torchvision
import os

root = '../project/data'
if not os.path.exists(root):
    os.makedirs(root)

print("Downloading CIFAR-10...")
torchvision.datasets.CIFAR10(root=root, train=True, download=True)
torchvision.datasets.CIFAR10(root=root, train=False, download=True)
print("Done.")
