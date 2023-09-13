import numpy as np
from keras.layers import LSTM, Dense

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


class chunk_LSTM():
    def __init__(self, n_lstm,chunk_size):
        self.layer_list = []
        for lstm_idx in range(n_lstm):
            self.layer_list.append(LSTM(1, input_shape=(1,chunk_size), activation='tanh', return_state=True))
        self.n_lstm = n_lstm
        self.chunk_size=chunk_size

    def predict(self, config):
        n_lstm = self.n_lstm
        chunk_size=self.chunk_size
        config_size = len(config)
        lstm_idx = 0
        output_list = []
        config_size=len(config)
        config_idx=0
        while config_idx < config_size:
            current_chunk=config[config_idx:config_idx+chunk_size]
            while len(current_chunk)<chunk_size :
                current_chunk +=[0]
            elem = np.array(current_chunk).reshape((1,1,chunk_size)).astype(np.float32)
            if config_idx == 0:
                output, state_h, state_c = self.layer_list[lstm_idx](elem)
            else:
                output, state_h, state_c = self.layer_list[lstm_idx](elem, initial_state=encoder_state)
            config_idx += chunk_size
            encoder_state = [state_h, state_c]
            output_list.append(output.numpy()[0][0])
#             set_trace()
            lstm_idx += 1
            if lstm_idx >= n_lstm:
                lstm_idx = 0 # Circular
        output_list = output_list[-n_lstm:]
        if len(output_list) < n_lstm:
            output_list += [0.0] * (n_lstm - len(output_list))
        return output_list

    def lstmX(self, configs):
        xdata = []
        for x in configs:
            x = self.predict(x)
            x = list(np.reshape(x, (self.n_lstm,)))
            xdata.append(x)
        xdata = np.array(xdata, dtype=np.double)
        return xdata
    def lstmY(self, fitnesses):
        ydata = []
        for f in fitnesses:
            ydata.append(f)
        ydata = np.array(ydata, dtype=np.double)
        return ydata

    def convertTrainData(self, chromosomes):
        configs=[]
        fitnesses=[]
        for chromosome in chromosomes:
            configs.append(chromosome.config)
            fitnesses.append(chromosome.fitness)
        xdata =self.lstmX(configs)
        ydata=self.lstmY(fitnesses)
        return xdata, ydata