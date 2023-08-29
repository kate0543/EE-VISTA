import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
    
    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_step, hidden, cell):
        embedded = self.embedding(input_step)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.out(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        target_len = target.shape[0]
        batch_size = target.shape[1]
        target_vocab_size = self.decoder.out.out_features
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        hidden, cell = self.encoder(source)
        
        input_step = target[0, :]
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(input_step, hidden, cell)
            outputs[t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2)
            input_step = target[t] if teacher_force else top1
        
        return outputs

# Example usage
input_size = 10  # Example vocabulary size
output_size = 10  # Example vocabulary size
hidden_size = 256
encoder = Encoder(input_size, hidden_size)
decoder = Decoder(hidden_size, output_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(encoder, decoder, device).to(device)

# Generate dummy input and target
source_seq = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(device)  # Variable length
target_seq = torch.tensor([[7, 8, 9], [1, 2, 3]]).to(device)  # Variable length
output = model(source_seq, target_seq)
print(output.shape)  # Example: torch.Size([target_len, batch_size, target_vocab_size])
