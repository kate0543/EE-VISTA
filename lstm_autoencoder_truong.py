import numpy as np
from numpy import array
from pprint import pprint
import copy
import random
from time import time

from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model

from pdb import set_trace

FIXED_LENGTH_REPRESENTATION_SIZE = 7

def create_variable_length_data(n_inst):
    data = []
    for inst_idx in range(n_inst):
        n_timestep = np.random.randint(1, high=10)
        seq = np.random.rand(n_timestep).tolist()
        data.append(seq)
    return data

def build_base_lstm_model(n_timestep):   
    # wrap encoderdecoder together
    t1 = time()
    model = Sequential()

    # encoder
    model.add(LSTM(FIXED_LENGTH_REPRESENTATION_SIZE, activation='relu', input_shape=(n_timestep,1)))
    model.add(RepeatVector(n_timestep))
    # decoder
    model.add(LSTM(FIXED_LENGTH_REPRESENTATION_SIZE, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))

    model.compile(optimizer='adam', loss='mse')
    t2 = time()
    print(t2-t1)
    return model

class VariableLengthAutoencoderLSTM:
    def __init__(self):
        pass

    def train(self, var_len_seq, n_epoch):
        print("Train the variable-length LSTM autoencoder!")
        shuffled_var_len_seq = copy.deepcopy(var_len_seq)
        curr_weights = None
        for epoch_idx in range(n_epoch):
            print("Epoch: %d" % (epoch_idx+1))
            random.shuffle(shuffled_var_len_seq)
            for seq in shuffled_var_len_seq:
                # reshape input into [samples, timesteps, features]
                n_timestep = len(seq)
                seq = np.array(seq)
                seq = seq.reshape((1,  n_timestep, 1))
                if curr_weights is None:
                    model = build_base_lstm_model(n_timestep)
                    model.fit(seq, seq, epochs=2)
                    
                    # First data instance, we train from scratch then save weights to curr_weights
                    curr_weights = model.get_weights()            

                else:
                    model = build_base_lstm_model(n_timestep)
                    t1 = time()
                    # Get weights from current model trained in previous instance. 
                    # This helps with ensuring that our model can handle variable-length inputs
                    model.set_weights(curr_weights) 
                    t2 = time()
                    print("ABC")
                    print(t2-t1)
                    
                    model.fit(seq, seq, epochs=2)
                    
                    t1 = time()
                    # Save the current weights
                    curr_weights = model.get_weights()     
                    t2 = time()
                    print("DEF")
                    print(t2-t1)
            
            
        self.model = model
    
    def encode(self, var_len_seq):
        print("Encode into fixed length representation after training!")
        print(var_len_seq)
        fixed_length_outputs  = []
        set_trace()
        for seq in var_len_seq:
            n_timestep = len(seq)
            seq = np.array(seq)
            seq = seq.reshape((1,  n_timestep, 1))
            
            model = build_base_lstm_model(n_timestep)
            curr_weights = self.model.get_weights()   
            model.set_weights(curr_weights) 
            
            encoder_only_model = Model(inputs=model.inputs, outputs=model.layers[0].output)
            fixed_length_output = encoder_only_model.predict(seq)
            fixed_length_output = fixed_length_output.tolist()
            fixed_length_output = fixed_length_output[0]
            
            fixed_length_outputs.append(fixed_length_output)
        
        return fixed_length_outputs
            
            
def main():
    n_inst = 5
    var_len_seq = create_variable_length_data(n_inst)
    print("Encode into fixed length representation before training!")
    pprint(var_len_seq)
    
    var_len_autoencoder_lstm_model = VariableLengthAutoencoderLSTM()
    var_len_autoencoder_lstm_model.train(var_len_seq, 2)
    out = var_len_autoencoder_lstm_model.encode(var_len_seq)
    
    print("Fixed-length output from trained variable-length autoencoder LSTM:")
    pprint(out)
    
    
if __name__ == "__main__":
    main()