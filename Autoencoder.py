# -*- coding: utf-8 -*-
##########################
## Autoencoder CIFAR-10 ##
##########################

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
from torch.autograd import Variable


config = dict(
  num_epochs = 5,
  batch_size = 32,
  weight_decay = 1e-5,
  )

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = tv.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)

dataloader = torch.utils.data.DataLoader(trainset, batch_size = config['batch_size'], shuffle = False, num_workers = 4)


# Autoencoder class
class Autoencoder(nn.Module):
  def __init__(self):
    super(Autoencoder, self).__init__()
    # Encoder
    self.encoder = nn.Sequential(
        nn.Conv2d(3,6, kernel_size = 5),
        nn.ReLU(True),
        nn.Conv2d(6,16, kernel_size = 5),
        nn.ReLU(True)
    )
    # Decoder
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(16,6, kernel_size = 5),
        nn.ReLU(True),
        nn.ConvTranspose2d(6,3, kernel_size = 5),
        nn.ReLU(True)
    )
    
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

model = Autoencoder().cpu()
distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay = config['weight_decay'])

losses = []
for epoch in range(config['num_epochs']):
    for data in dataloader:
        img, _ = data
        img = Variable(img).cpu()
        # Forward step
        output = model(img)
        loss = distance(output, img)
        losses.append(loss.item())
        # Backward step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, config['num_epochs'], loss.item()))

import matplotlib.pyplot as plt
plt.plot(losses)

