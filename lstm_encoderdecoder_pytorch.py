__author__ = 'SherlockLiao'

import os
import random
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

# if not os.path.exists('./mlp_img'):
#     os.mkdir('./mlp_img')


# def to_img(x):
#     x = 0.5 * (x + 1)
#     x = x.clamp(0, 1)
#     x = x.view(x.size(0), 1, 28, 28)
#     return x

n_inst = 5
FIXED_LENGTH_REPRESENTATION_SIZE = 7
num_epochs = 100
batch_size = 128
learning_rate = 1e-3

# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])

# dataset = MNIST('./data', transform=img_transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



def create_variable_length_data(n_inst):
    # data = []
    # Generate random dimensions for the tensor
    num_dims = n_inst
    # random.randint(2, 4)  # Choose a random number of dimensions between 2 and 4
    dims = [random.randint(3, 6) for _ in range(num_dims)]  # Generate random sizes for each dimension

    # Create a random tensor with the generated dimensions
    random_tensor = torch.rand(*dims)

    # print("Random Tensor:")
    # print(random_tensor)
    print("Tensor Shape:", random_tensor.shape)
    return random_tensor

dataloader=[]
for i in range (0,10):
    var_len_seq = create_variable_length_data(n_inst)
    data=var_len_seq
    dataloader.append(data)

class Autoencoder(nn.Module):
    def __init__(self, latent_size):
        super(Autoencoder, self).__init__()
        input_size=n_inst
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, latent_size))
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_size),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = Autoencoder(latent_size=FIXED_LENGTH_REPRESENTATION_SIZE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam( model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        # img, _ = data
        # img = img.view(img.size(0), -1)
        # img = Variable(img).cuda()
        # ===================forward=====================
        output = model(data)
        loss = criterion(output, data)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data[0]))
    # if epoch % 10 == 0:
    #     pic = to_img(output.cpu().data)
    #     save_image(pic, './mlp_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './sim_autoencoder.pth')


