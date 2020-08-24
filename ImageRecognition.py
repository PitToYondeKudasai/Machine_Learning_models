## Image Recognition ##
'''
This model use a simple CNN trained on the MNIST dataset
'''
import torch
import torch.nn as nn
import torchvision
import numpy as np
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

## -- CNN -- ##
class ImageRecognizer(nn.Module):
  def __init__(self, n_channel):
    super().__init__()
    self.conv1 = nn.Conv2d(n_channel, 5, kernel_size = 3)
    self.conv2 = nn.Conv2d(5, 5, kernel_size = 3)
    self.fc1 = nn.Linear(5 * 24 * 24, 1000)
    self.fc2 = nn.Linear(1000, 10)
    self.soft = nn.Softmax(1)

  def forward(self, obs):
    out = self.conv1(obs)
    out = self.conv2(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc1(out)
    out = self.fc2(out)
    return self.soft(out)

# We create the image recognizer NN
image_rec = ImageRecognizer(1)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(image_rec.parameters(), lr = 0.001, momentum = 0.9)

# We load the training dataset
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size = 6, shuffle=True)

## -- Training -- ##
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = image_rec(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

## -- Testing -- ##
# We load the test dataset
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size = 6, shuffle=True)
examples = enumerate(test_loader)

batch_idx, (example_data, example_targets) = next(examples)
test = image_rec.forward(example_data)
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    test.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])

