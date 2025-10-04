# data.py
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# CIFAR-10 has 3 channels. Using single-value mean/std applies to all channels.
def get_loaders(batch_size_train=128, batch_size_test=100, img_size=96, root="./data", num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # maps to [-1, 1] per channel
    ])

    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    testset  = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    testloader  = DataLoader(testset,  batch_size=batch_size_test,  shuffle=False, num_workers=num_workers)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    return trainloader, testloader, classes
