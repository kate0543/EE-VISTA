import random as rand

import numpy as np
from smt.surrogate_models import RBF, IDW, LS, KRG, KPLS, KPLSK, MGP

from surrogate.lstm_autoencoder import VariableLengthAutoencoderLSTM;
from surrogate.chunk_lstm import chunk_LSTM

class smtWrapper:
    def __init__(self,n_lstm,chunk_size):
        # add lstm layer to convert variable length binary encodings
        # self.n_lstm=n_lstm
        # self.chunk_size=chunk_size
        # self.lstm = chunk_LSTM(self.n_lstm,self.chunk_size)
        self.n_epoch=10
        self.var_len_autoencoder_lstm_model=VariableLengthAutoencoderLSTM()


        # add a list of regressors from smt toolbox
        self.regressors = []
        self.regressors.append(
            RBF(d0=5, print_solver=False, print_training=False, print_prediction=False, print_problem=False))

    def initTrainData(self, pop):
        var_len_seq=[]
        fitnesses=[]
        for individual in pop:
            var_len_seq.append(individual.config)
            fitnesses.append(individual.fitness)
        self.var_len_autoencoder_lstm_model.train(var_len_seq,self.n_epoch)
        out = self.var_len_autoencoder_lstm_model.encode(var_len_seq)
        # convert initial train database on passed population
        # self.xdata, self.ydata = self.lstm.convertTrainData(pop)
        self.xdata=np.array(out, dtype=np.double)
        self.ydata=fitnesses
        # remove duplicate x,y dataset
        self.drop_duplicates()

    def fit(self):
        # set up data for regressors
        for regressor in self.regressors:
            regressor.set_training_values(
                np.array(self.xdata, dtype=np.double), np.array(self.ydata, dtype=np.double))

    def train(self):
        # train regressors
        for regressor in self.regressors:
            regressor.train()

    def predict(self, x):
        # predict fitness, use average value
        fitness = []
        for regressor in self.regressors:
            encodings=self.var_len_autoencoder_lstm_model.encode(x) 
            fitness.append(regressor.predict_values(np.array(encodings,dtype=np.double)).ravel())
# =        return sum(fitness) / len(fitness)
        output=sum(fitness) / len(fitness)
        print(output)
        return output

    def predict_weight(self, x):
        # predict fitness, use average value
        fitness = []
        scores=[]
        for regressor in self.regressors:
            encodings=self.var_len_autoencoder_lstm_model.encode(x) 
            fitness.append(regressor.predict_values(np.array(encodings,dtype=np.double)).ravel())
        for regressor in self.regressors:
            regressor.score()
        # return sum(fitness) / len(fitness)
        output=sum(fitness) / len(fitness)
        print(output)
        return output
 

    def convertTrainData(self, pop):
        var_len_seq=[]
        fitnesses=[]
        for individual in pop:
            var_len_seq.append(individual.config)
            fitnesses.append(individual.fitness)
        self.var_len_autoencoder_lstm_model.train(var_len_seq,self.n_epoch)
        out = self.var_len_autoencoder_lstm_model.encode(var_len_seq) 
        xdata=np.array(out, dtype=np.double)
        ydata=fitnesses
        # remove duplicate x,y dataset
        self.drop_duplicates()
        return xdata, ydata
    
    def updateData(self, pop):
        # update train data with new received population
        print('update smt train data....')
        xnew, ynew = self.convertTrainData(pop)
        self.xdata = np.append(self.xdata, xnew, axis=0)
        self.ydata = np.append(self.ydata, ynew, axis=0)
        self.drop_duplicates()

    def updateModel(self):
        # retrain regressors
        print('retrain smt models')
        self.fit()
        self.train()

    def selection(self, pop, mode):
        # select invidiuals for real fitness evaluation and update regressors database
        print(mode, ' selection for smt update')
        subpop = []
        if mode == 'random':
            size = int(len(pop) * 0.3)
            subpop = rand.choices(pop, k=size)
        elif mode == 'FP':
            size = int(len(pop) * 0.3)
            pop_fitness = np.array([c.get_fitness() for c in pop])
            subpop = rand.choices(pop, k=size, weights=pop_fitness)
        elif mode == 'best':
            subpop = pop[:1]
            print(len(subpop))
        return subpop

    def drop_duplicates(self):
        # remove duplicate (x,y)
        data = set()
        for x, y in zip(self.xdata, self.ydata):
            data.add((tuple(x), y))
        X = []
        Y = []
        for x, y in data:
            X.append(x)
            Y.append(y)
        self.xdata = np.array(X, dtype=np.double)
        self.ydata = np.array(Y, dtype=np.double)
