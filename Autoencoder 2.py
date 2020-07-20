#################
## Autoencoder ##
#################
# This code implements a very simple example of autoencoder

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, data):
        x = self.linear1(data)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = torch.tanh(x)
        return x

class Decoder(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, data):
        x = self.linear1(data)
        x = torch.tanh(x)
        x = self.linear2(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, output_size)
        self.decoder = Decoder(output_size, hidden_size, input_size)
        
    def forward(self, data):
        h = self.encoder(data)
        out =  self.decoder(h)
        return out

#-------------------------#
# We create the training set
x = np.random.rand(100,1)
X = np.concatenate((x,2*x,3*x,4*x,5*x),axis=1)
X = torch.tensor(X, dtype = torch.float)

batch_size, input_size = X.shape
output_size = 1
hidden_size = 10

# We initialize the Autoencoder
AC = Autoencoder(input_size, hidden_size, output_size)

criterion = torch.nn.MSELoss(reduction = 'mean')
AC_optimizer = torch.optim.Adam(AC.parameters(), lr = 1e-4)

for i in range(10000):
    X_pred = AC(X)
    loss = criterion(X_pred, X)
    AC_optimizer.zero_grad()
    loss.backward()
    AC_optimizer.step()
    if i % 1000 == 0:
        print(i,loss.item())

# We test the model
x_test = np.random.rand(5,1)
X_test = np.concatenate((x_test,2*x_test,3*x_test,4*x_test,5*x_test),axis=1)
X_test = torch.tensor(X_test, dtype = torch.float)
X_test_pred = AC(X_test)

print("Test set:")
print(X_test)
print("Reconstruction:")
print(X_test_pred)


        
        
