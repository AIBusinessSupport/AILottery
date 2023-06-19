from keras.layers import Dense, Input, Reshape, Conv2D, ReLU, BatchNormalization, Flatten, AveragePooling2D, Add, LSTM, ELU
from keras.models import Model, Sequential


def FFNN(neurons):
	inputs = Input((neurons[0],))
	for ind, neuron in enumerate(neurons[1:-1]):
		x = Dense(neuron, activation='relu')(x if ind > 0 else inputs)
	x = Dense(20, activation='elu')(x)
	return Model(inputs=inputs, outputs=x)

def CNN(params):
	inputs = Input((params[0], 20))
	x = Reshape((params[0], 20, 1))(inputs)
	x = Conv2D(params[1], kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(x)
	x = BatchNormalization()(x)
	x = ReLU()(x)
	x = AveragePooling2D((2, 2))(x)
	x_ = Conv2D(params[2], kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(x)
	x = BatchNormalization()(x_)
	x = ReLU()(x)
	x = Conv2D(params[3], kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(x)
	if params[2] == params[3]: x = Add()([x, x_])
	x = BatchNormalization()(x)
	x = ELU()(x)
	x = AveragePooling2D((2, 2))(x)
	x = Flatten()(x)
	x = Dense(params[4], activation='elu')(x)
	x = Dense(20, activation='tanh')(x)
	return Model(inputs=inputs, outputs=x)
##
def RNN(params):
	model = Sequential()
	model.add(LSTM(params[1], return_sequences=True, input_shape=(params[0], 20)))
	for param in params[2:-2]:
		model.add(LSTM(param, return_sequences=True))
	model.add(LSTM(params[-2]))
#    model.add(Dense(params[-1], activation=None))
	model.add(Dense(params[-1], activation='relu'))
    
	return model
