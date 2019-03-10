import numpy as np
import os
import matplotlib.pyplot as plt
import struct
import sys
import math


def loadMNIST(dataset = "training", path = "."):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = np.fromfile(flbl, dtype=np.int8)
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows*cols)
    fimg.close()

    return img, lbl

def one_hot(data):
    data_size =  data.shape[0]
    encode_data = np.zeros( (data_size,10) )
    for i,t in enumerate(data):
        encode_data[i,t] = 1
    return encode_data 
        
class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x < 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
         
def softmax(x):
    x = x.T    
    x = x - np.max(x, axis=0)
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return y.T 

def cross_entropy_error(y, t):
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y+1e-10))  / batch_size
    #return -np.sum(t * np.log(y)) / batch_size
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, t)
        return self.loss

    #def backward(self, dout = 1):
    #    batch_size = self.t.shape[0]
    #    dx = (self.y - self.t) / batch_size
    #    return dx

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: 
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx    
        
class Affine:
    def __init__(self, W, b, activation=None):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        self.activation = activation

    def forward(self, x):
        self.x = x
        #print(self.x.shape, self.W.shape,self.b.shape)
        out = np.dot(self.x, self.W) + self.b
        if(self.activation):
            return self.activation.forward(out)
        else:
            return out
        
    def backward(self, dout):
        if(self.activation):
            dout2 = self.activation.backward(dout)
        else:
            dout2 = dout
            
        dx = np.dot(dout2, self.W.T)
        self.dW = np.dot(self.x.T, dout2)
        self.db = np.sum(dout2, axis=0)
        return dx

class Modle:
    def __init__(self, input_size, learning_rate = 0.01, weight_init_std = 0.01, lastLayer = SoftmaxWithLoss()):
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.layers = []
        self.layers_size = [input_size,]
        self.weight_init_std = weight_init_std
        self.lastLayer = lastLayer

        
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
        
    def predict2(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return softmax(x)
        
    def loss(self, x, t):
        y = self.predict(x)
        loss = self.lastLayer.forward(y, t) #SoftmaxWithLoss
        return loss
        
    def add_layer(self, node_num, activation=None):
        w = self.weight_init_std * np.random.randn(self.layers_size[-1], node_num)
        #b = np.zeros(node_num)
        b = self.weight_init_std * np.random.randn(node_num)
        
        self.layers.append( Affine(w, b, activation) )
        self.layers_size.append(node_num)
        return
        
    def train(self, x, t):
        # forward
        loss = self.loss(x, t)
        
        # backward
        dout = self.lastLayer.backward(1)
        #print("dout",dout)
        
        #rev_layers = self.layers.copy()
        #rev_layers.reverse()
        
        self.layers.reverse()
        for i,layer in enumerate(self.layers):
            #print(i,layer.dW)
            dout = layer.backward(dout)
            layer.W -= self.learning_rate * layer.dW
            layer.b -= self.learning_rate * layer.db
        self.layers.reverse()
            
        return 
    
    def accuracy(self, x, t):
        y = self.predict2(x)
        y = np.argmax(y, axis=1)
        
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
def draw(data,type,file_name):
    x = data
    #y = range(0,len(data))
    plt.figure()
    plt.plot(x)
    plt.title(file_name[0:-9])
    plt.xlabel('epoch')
    #plt.ylabel('I am y')

    if (type == "a"):
        file_dir = "result/1/a_wide_hidden_layer/"
    elif(type == "b"):
        file_dir = "result/1/b_deep_hidden_layer/"
    else:    
        print("saving type error, failed to save results ")
        return
    if not(os.path.isdir(file_dir)):
        os.makedirs(file_dir)
    plt.savefig(file_dir + file_name)
    plt.show()

def save_model(model, file_dir, file_name):
    if not(os.path.isdir(file_dir)):
        os.makedirs(file_dir)
    with open(file_dir+file_name,'wb') as model_f:
        pickle.dump(model, model_f)
    
def load_model(file_name):
    with open(file_name,'rb') as model_f:
        model = pickle.load(model_f)
    return model
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b',
                        default=64,
                        type=int,
                        dest='BATCH_SIZE',
                        help='batch size,(default=64)')
    parser.add_argument('-lr',
                        default=0.1,
                        type=float,
                        dest='LEARNING_RATE',
                        help='learning rate,(default=0.1)')                        
    parser.add_argument('-ep',
                        default=20,
                        type=int,
                        dest='EPOCH',
                        help='epoch number,(default=20)')
    args = parser.parse_args()
    BATCH_SIZE = args.BATCH_SIZE
    LEARNING_RATE = args.LEARNING_RATE
    EPOCH = args.EPOCH
    print("batch size:",BATCH_SIZE)
    print("learning rate:",LEARNING_RATE)
    print("epoch num:",EPOCH)
    print()
    
    x_train, y_train = loadMNIST(dataset="training", path="MNIST_data")
    x_test, y_test = loadMNIST(dataset="testing", path="MNIST_data")
    
    x_train = x_train/255
    x_test = x_test/255
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)
    
    epoch = 20
    train_size = x_train.shape[0]
    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE
    iter_per_epoch = math.ceil(max(train_size / batch_size, 1))
    iters_num = int(iter_per_epoch * epoch)
    
    #1.(a)
    print("1.(a) wide hidden layer")
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    network = Modle(input_size=784, learning_rate = learning_rate, lastLayer = SoftmaxWithLoss())
    network.add_layer(256,ReLU())
    network.add_layer(10)       
    
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = y_train[batch_mask]
        
        network.train(x_batch, t_batch)
        
        if i % iter_per_epoch == 0:
            train_loss = network.loss(x_train, y_train)
            test_loss = network.loss(x_test, y_test)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            
            train_acc = network.accuracy(x_train, y_train)
            test_acc = network.accuracy(x_test, y_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train_acc:",train_acc, "test_acc:", test_acc)
    
    draw(train_loss_list,"a","train_loss_list.png")
    draw(test_loss_list,"a","test_loss_list.png")
    draw(train_acc_list,"a","train_acc_list.png")
    draw(test_acc_list,"a","test_acc_list.png")
    
    #1.(b)
    print()
    print("1.(b) deep hidden layer")
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
        
    network2 = Modle(input_size=784, learning_rate = learning_rate, lastLayer = SoftmaxWithLoss())
    network2.add_layer(204,ReLU())
    network2.add_layer(204,ReLU())
    network2.add_layer(10)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = y_train[batch_mask]
        
        network2.train(x_batch, t_batch)
        
        if i % iter_per_epoch == 0:
            train_loss = network2.loss(x_train, y_train)
            test_loss = network2.loss(x_test, y_test)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            
            train_acc = network2.accuracy(x_train, y_train)
            test_acc = network2.accuracy(x_test, y_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train_acc:",train_acc, "test_acc:", test_acc)
    
    draw(train_loss_list,"b","train_loss_list.png")
    draw(test_loss_list,"b","test_loss_list.png")
    draw(train_acc_list,"b","train_acc_list.png")
    draw(test_acc_list,"b","test_acc_list.png")
    
    
    