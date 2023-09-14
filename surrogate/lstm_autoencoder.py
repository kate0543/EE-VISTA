import numpy as np 
import copy
import random 

from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# from pdb import set_trace

# output size for encoder
FIXED_LENGTH_REPRESENTATION_SIZE = 10

# def create_variable_length_data(n_inst):
#     data = []
#     for inst_idx in range(n_inst):
#         n_timestep = np.random.randint(1, high=10)
#         seq = np.random.rand(n_timestep).tolist()
#         print(len(seq))
#         data.append(seq)
#     return data

def build_base_lstm_model(n_timestep):   
    # wrap encoderdecoder together 
    model = Sequential()

    # encoder
    model.add(LSTM(FIXED_LENGTH_REPRESENTATION_SIZE, activation='relu', input_shape=(n_timestep,1)))
    model.add(RepeatVector(n_timestep))
    # decoder
    model.add(LSTM(FIXED_LENGTH_REPRESENTATION_SIZE, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))

    model.compile(optimizer='adam', loss='mse') 
    return model

class VariableLengthAutoencoderLSTM:
    def __init__(self):
        pass

    def train(self, var_len_seq, n_epoch):
        print("Train the variable-length LSTM autoencoder!")
        shuffled_var_len_seq = copy.deepcopy(var_len_seq)
        curr_weights = None
        for epoch_idx in range(n_epoch):
            # print("Epoch: %d" % (epoch_idx+1))
            random.shuffle(shuffled_var_len_seq)
            for seq in shuffled_var_len_seq:
                # reshape input into [samples, timesteps, features]
                n_timestep = len(seq)
                seq = np.array(seq)
                seq = seq.reshape((1,  n_timestep, 1))
                if curr_weights is None:
                    model = build_base_lstm_model(n_timestep)
                    model.fit(seq, seq, epochs=n_epoch,verbose=0)
                    
                    # First data instance, we train from scratch then save weights to curr_weights
                    curr_weights = model.get_weights()            

                else:
                    model = build_base_lstm_model(n_timestep) 
                    # Get weights from current model trained in previous instance. 
                    # This helps with ensuring that our model can handle variable-length inputs
                    model.set_weights(curr_weights)   
                    model.fit(seq, seq, epochs=n_epoch,verbose=0) 
                    # Save the current weights
                    curr_weights = model.get_weights()      
            
            
        self.model = model
    
    def encode(self, var_len_seq): 
        fixed_length_outputs  = [] 
        for seq in var_len_seq:
            n_timestep = len(seq)
            seq = np.array(seq)
            seq = seq.reshape((1,  n_timestep, 1))
            
            model = build_base_lstm_model(n_timestep)
            curr_weights = self.model.get_weights()   
            model.set_weights(curr_weights) 
            
            encoder_only_model = Model(inputs=model.inputs, outputs=model.layers[0].output)
            fixed_length_output = encoder_only_model.predict(seq, verbose = 0)
            fixed_length_output = fixed_length_output.tolist()
            fixed_length_output = fixed_length_output[0]
            
            fixed_length_outputs.append(fixed_length_output)
        
        return fixed_length_outputs


def predict(var_len_seq,n_epoch):               
    var_len_autoencoder_lstm_model = VariableLengthAutoencoderLSTM()
    var_len_autoencoder_lstm_model.train(var_len_seq, n_epoch)
    out = var_len_autoencoder_lstm_model.encode(var_len_seq)
    print("Fixed-length output from trained variable-length autoencoder LSTM:")
    print(out)
    return out
