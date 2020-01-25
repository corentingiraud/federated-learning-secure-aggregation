import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import numpy as np
from pandas import read_csv
from functools import reduce

# Use eager execution to access .numpy() https://www.tensorflow.org/guide/eager
tf.executing_eagerly()

# MODEL params

verbose = 0
epochs = 500
batch_size = 500
learning_rate = 0.0004
standar = False
activation_conv1D = 'relu'
filters_conv1D = 8
kernel_conv1D = 3
dropout_ratio = 0.5
pool_size_maxpool = 3
activation_dense1 = 'relu'
dense1_neurons = 20
activation_dense2 = 'softmax'
optimizer = 'adam'
loss = 'mean_squared_error'

param = [verbose, epochs, batch_size, learning_rate, dropout_ratio, standar, activation_conv1D, activation_dense1, activation_dense2, filters_conv1D, kernel_conv1D, pool_size_maxpool, dense1_neurons, optimizer, loss]

class ModelTf():
    def __init__(self):
        self.testX, self.testY = self.load_dataset_test()

        n_timesteps, n_features, n_outputs = 128, 6, 6

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv1D(filters=int(param[9]), kernel_size=int(param[10]), activation=param[6], input_shape=(n_timesteps,n_features)))
        #model.add(Conv1D(filters=8, kernel_size=5, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(float(param[4])))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=int(param[11])))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(int(param[12]), activation=param[7]))
        self.model.add(tf.keras.layers.Dense(n_outputs, activation=param[8]))

        #model.summary()
        self.model.compile(loss=param[14], optimizer=param[13], metrics=['accuracy'])

    def getTrainableVars(self):
        return self.model.trainable_variables

    def toNumpyFlatArray(self):
        flatList = []
        for trainableVar in self.getTrainableVars():
            if len(trainableVar.shape) >= 2:
                for item in trainableVar.read_value().numpy().flatten().tolist():
                    flatList.append(item)
            else:
                for item in trainableVar.read_value().numpy().tolist():
                    flatList.append(item)
        flatArray = np.asarray(flatList, dtype=np.dtype('d'))
        return flatArray

    def updateFromNumpyFlatArray(self, flatArray):
        index = 0
        for trainableVar in self.getTrainableVars():
            if len(trainableVar.shape) >= 2:
                array = flatArray[index:index + (reduce(lambda x, y: x * y, trainableVar.shape))]
                reshapedArray = np.reshape(array, trainableVar.shape)
                trainableVar.assign(reshapedArray)
                index += reduce(lambda x, y: x * y, trainableVar.shape)
            else:
                array = flatArray[index:index + trainableVar.shape[0]]
                reshapedArray = np.reshape(array, trainableVar.shape)
                trainableVar.assign(reshapedArray)
                index += trainableVar.shape[0]

    def trainModel(self, x_train, y_train):
        res = self.model.fit(x_train, y_train, epochs=5, verbose=0)
        return res.history['accuracy'][-1]

    def test(self):
        res = self.model.evaluate(self.testX,  self.testY, verbose=0)
        return res[1]

    # load a single file as a numpy array
    def load_file(self, filepath):
        dataframe = read_csv(filepath, header=None, delim_whitespace=True)
        return dataframe.values

    # load a list of files and return as a 3d numpy array
    def load_group(self, filenames, prefix=''):
        loaded = list()
        for name in filenames:
            data = self.load_file(prefix + name)
            loaded.append(data)
        # stack group so that features are the 3rd dimension
        loaded = np.dstack(loaded)
        return loaded

    # load a dataset group, such as train or test
    def load_dataset_group(self, group, prefix=''):
        filepath = prefix + group + '/Inertial Signals/'
        # load all 9 files as a single array
        filenames = list()
        # total acceleration
        #filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
        # body acceleration
        filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
        # body gyroscope
        filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
        # load input data
        X = self.load_group(filenames, filepath)
        # load class output
        y = self.load_file(prefix + group + '/y_'+group+'.txt')
        return X, y
    # load the dataset, returns train and test X and y elements
    def load_dataset(self, prefix=''):
        # load all train
        trainX, trainy = self.load_dataset_group('train', prefix + 'app/Motion2/')
        # load all test
        testX, testY = self.load_dataset_group('test', prefix + 'app/Motion2/')
        # zero-offset class values
        trainy = trainy - 1
        testY = testY - 1
        # one hot encode y
        trainy = tf.keras.utils.to_categorical(trainy)
        testY = tf.keras.utils.to_categorical(testY)
        return trainX, trainy, testX, testY
    # load the dataset, returns test X and y elements
    def load_dataset_test(self, prefix=''):
        # load all test
        testX, testY = self.load_dataset_group('test', prefix + 'app/Motion2/')
        # zero-offset class values
        testY = testY - 1
        # one hot encode y
        testY = tf.keras.utils.to_categorical(testY)
        return testX, testY

    def load_dataset_walk(self, prefix=''):
        # load all train
        trainX, trainy = self.load_dataset_group('train', prefix + 'app/Motion2/')
        # load all test
        testX, testY = self.load_dataset_group('test', prefix + 'app/Motion2/')

        # zero-offset class values
        trainy = trainy - 1
        testY = testY - 1
        for i in trainy:
            if i[0] != 0:
                i[0] = 1
        for i in testY:
            if i[0] != 0:
                i[0] = 1

        # one hot encode y
        trainy = tf.keras.utils.to_categorical(trainy)
        testY = tf.keras.utils.to_categorical(testY)
        return trainX, trainy, testX, testY
