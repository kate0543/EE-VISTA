import torch
import torch.nn as nn
from d2l import torch as d2l


encoder=d2l.Encoder
# Define the encoder
class Encoder(d2l.Encoder):
    def __init__(self, hidden_size=128):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, hidden_size, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return self.encoder(x)

# Define the decoder
class Decoder(d2l.Decoder):
    def __init__(self, hidden_size=128):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # To ensure output values are in [0, 1]
        )
    
    def forward(self, x):
        return self.decoder(x)

# Combine encoder and decoder to create the autoencoder
class EncoderDecoder(d2l.Classifier):
    def __init__(self, hidden_size=128):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(hidden_size)
        self.decoder = Decoder(hidden_size)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load and preprocess your dataset here
train_dl =100

# Initialize the autoencoder and other components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoderdecoder = EncoderDecoder().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(encoderdecoder.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    # for batch in train_dl:
    #     images = batch.to(device)
    X= torch.randn(3, 4)

    optimizer.zero_grad()

    outputs = encoderdecoder(X)
    loss = loss_fn(outputs, X)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model if needed
# torch.save(encoderdecoder.state_dict(), 'autoencoder.pth')

