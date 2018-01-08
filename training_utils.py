import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.models import load_model


def convert_output(alphabet):
    nn_outputs = []
    for alpha in alphabet:
        mask = np.zeros(10)
        index = alpha%10
        mask[index] = 1
        nn_outputs.append(mask)

    return np.array(nn_outputs)


def create_ann():
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(10, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)  # dati ulazi
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi za date ulaze

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=1500, batch_size=1, verbose=0, shuffle=False)

    return ann


def winner(output): # output je vektor sa izlaza neuronske mreze
    return max(enumerate(output), key=lambda x: x[1])[0]


def display_result(outputs, alphabet, real_res):
    result = []
    idx = 0
    count = 0

    for output in outputs:
        vv = get_as_dec(alphabet[winner(output)])
        result.append(vv)
        if vv!=real_res[idx]:
            print("=====\n")
            print(vv)
            print(" vs ")
            print(real_res[idx])
            print(output)
            count = count + 1
        idx = idx + 1

    print(count)
    print(idx)
    print((count*100)/idx)
    return result


def get_as_dec(arr):
    maxel = max(arr)

    for i in range(0, len(arr)):
        if arr[i]==maxel:
            return i

    return -1


def save_model(ann):
    ann.save('classifier.h5')


def load_modell():
    return load_model("classifier.h5")
