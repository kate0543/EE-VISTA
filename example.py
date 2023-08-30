import lstm_encoderdecoder_pytorch
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
import numpy as np
import random
import os, errno
import sys
from tqdm import trange

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

n_inst = 5
FIXED_LENGTH_REPRESENTATION_SIZE = 7
num_epochs = 100
batch_size = 128
learning_rate = 1e-3  

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

n_inst = 5
FIXED_LENGTH_REPRESENTATION_SIZE = 7
num_epochs = 100
batch_size = 128
learning_rate = 1e-3  

model = lstm_encoderdecoder_pytorch.lstm_seq2seq(input_size = n_inst, hidden_size = 15)
loss = model.train_model(X_train, Y_train, n_epochs = 50, target_len = ow, batch_size = 5, training_prediction = 'mixed_teacher_forcing', teacher_forcing_ratio = 0.6, learning_rate = 0.01, dynamic_tf = False)

# # plot predictions on train/test data
# plotting.plot_train_test_results(model, Xtrain, Ytrain, Xtest, Ytest)

# plt.close('all')