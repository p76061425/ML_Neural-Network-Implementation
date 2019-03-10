import numpy as np
import os
import matplotlib.pyplot as plt
import struct
import sys
import pickle
import os
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

    def forward(self, x, train_flg):
        self.mask = x < 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

def cross_entropy_error(y, t):
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y+1e-10) + (1-t)*np.log((1-y)+1e-10))  / batch_size
    #return -np.sum(t * np.log(y)) / batch_size

def softmax(x):
    x = x.T    
    x = x - np.max(x, axis=0)
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return y.T 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
        
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
        
class SigmoidWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = sigmoid(x)
        self.loss = cross_entropy_error(self.y, t)
        return self.loss

    #def backward(self, dout = 1):
    #    batch_size = self.t.shape[0]
    #    dx = (self.y - self.t) / batch_size
    #    return dx

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # if self.t.size == self.y.size: 
        #     dx = (self.y - self.t) / batch_size
        # else:
        #     dx = self.y.copy()
        #     dx[np.arange(batch_size), self.t] -= 1
        #     dx = dx / batch_size
        return (self.y - self.t) / batch_size        
        
class Affine:
    def __init__(self, W, b, activation=None):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        self.activation = activation

    def forward(self, x, train_flg):
        self.x = x
        #print(self.x.shape, self.W.shape,self.b.shape)
        out = np.dot(self.x, self.W) + self.b
        if(self.activation):
            return self.activation.forward(out,train_flg)
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
        
class Dropout:

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.W       = np.array([])
        self.b       = np.array([])
        self.dW      = np.array([])
        self.db      = np.array([])

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask 
 
        
class Modle:
    def __init__(self, input_size, learning_rate = 0.01, weight_init_std = 0.01, lastLayer = SigmoidWithLoss()):
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.layers = []
        self.layers_size = [input_size,]
        self.weight_init_std = weight_init_std
        self.lastLayer = lastLayer
        #self.train_loss_list = [] 
        
    def predict(self, x, train_flg):
        for layer in self.layers:
            x = layer.forward(x,train_flg)
        return x
        
    def predict2(self, x, train_flg):
        for layer in self.layers:
            x = layer.forward(x, train_flg)
        return sigmoid(x)
        
    def loss(self, x, t, train_flg):
        y = self.predict(x, train_flg)
        loss = self.lastLayer.forward(y, t)
        return loss
        
    def add_layer(self, node_num, activation=None, dropout=False,dropout_rate=0.5):
        w = self.weight_init_std * np.random.randn(self.layers_size[-1], node_num)
        #b = np.zeros(node_num)
        b = self.weight_init_std * np.random.randn(node_num)
        
        self.layers.append( Affine(w, b, activation) )
        self.layers_size.append(node_num)
        if(dropout):
            self.layers.append( Dropout(dropout_ratio=dropout_rate) )
            return
        else:
            return
            
    def train(self, x, t, denoise, train_flg):
        if(denoise):
            noise = 0.8
            _x = x + noise * (np.random.randn(x.size).reshape(x.shape))            
        else:
            _x = x
        # forward
        loss = self.loss(_x, t, train_flg)
        
        # backward
        dout = self.lastLayer.backward(1)
        
        self.layers.reverse()
        for i,layer in enumerate(self.layers):
            dout = layer.backward(dout)
            layer.W -= self.learning_rate * layer.dW
            layer.b -= self.learning_rate * layer.db
        self.layers.reverse()
            
        #self.train_loss_list.append(loss)
        return 
    
    def accuracy(self, x, t, train_flg):
        y = self.predict2(x, train_flg)
        y = np.argmax(y, axis=1)
        
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def dimension_reduction(self, x, dimension):
        for layer in self.layers:
            x = layer.forward(x,train_flg=False)
            if(x.shape[1] == dimension):
                return x
        print("network dimension error!")
                
def visualize_filters(img_data):
    fig = plt.figure()
    for i in range(16):
        fig.add_subplot(4,4,i+1)    
        plt.imshow(img_data[i].reshape([28,28]), cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(hspace=0.2,wspace=0.2)
    
    result_dir = "./result/2/"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    file_name = 'Visualize_filters.png'   
    plt.axis('equal')
    plt.savefig(result_dir + file_name)
    print("save fig",result_dir + file_name)
    
    plt.show()

def save_model(model, file_dir, file_name):
    if not(os.path.isdir(file_dir)):
        os.makedirs(file_dir)
    with open(file_dir+file_name,'wb') as model_f:
        pickle.dump(model, model_f)
    
def load_model(file_path):
    with open(file_path,'rb') as model_f:
        model = pickle.load(model_f)
    return model

#def draw_compare(img_1, img_2):
#  img_1 = img_1.reshape([28,28])
#  img_2 = img_2.reshape([28,28])
#
#  fig = plt.figure()
#  fig.add_subplot(1,2,1)
#  plt.title("Original Test Img")
#  plt.imshow(img_1, cmap='gray')
#  plt.axis('off')
#  fig.add_subplot(1,2,2)
#  plt.title("Reconstruct Test Img")
#  plt.imshow(img_2, cmap='gray')
#  plt.axis('off')
#  plt.subplots_adjust(hspace=0.2,wspace=0.2)
#  #file_name = "reconstruct_img" + ".png"
#  #plt.savefig(result_dir + file_name)
#  plt.show()

def draw_compare(org_img, reconstruct_img):
    
    orgImg_list = []
    for i in range(org_img.shape[0]):
        orgImg = org_img[i].reshape([28,28])
        orgImg_list.append(orgImg)
        
    reconImg_list = []
    for i in range(reconstruct_img.shape[0]):
        reconImg = reconstruct_img[i].reshape([28,28])
        reconImg_list.append(reconImg)

    fig = plt.figure()
    for i in range(0,4):
        fig.add_subplot(4,4,1+i)
        plt.imshow(orgImg_list[i], cmap='gray')
        plt.axis('off')
    for i in range(0,4):
        fig.add_subplot(4,4,5+i)
        plt.imshow(reconImg_list[i], cmap='gray')
        plt.axis('off')
    for i in range(4,8):
        fig.add_subplot(4,4,5+i)
        plt.imshow(orgImg_list[i], cmap='gray')
        plt.axis('off')  
    for i in range(4,8):
        fig.add_subplot(4,4,9+i)
        plt.imshow(reconImg_list[i], cmap='gray')
        plt.axis('off')

    plt.subplots_adjust(hspace=0.2,wspace=0.2)
    
    result_dir = "./result/2/"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    file_name = 'reconstruct_comparison.png'   
    plt.axis('equal')
    plt.savefig(result_dir + file_name)
    print("save fig",result_dir + file_name)
    
    plt.show()  
  
def plot_grid_scatter(x_train_org,dr_result,GRID_SIZE,dm_0,dm_1,IMG_SHOW):

    x_min = min(dr_result[:,dm_0])
    x_max = max(dr_result[:,dm_0])
    y_min = min(dr_result[:,dm_1])
    y_max = max(dr_result[:,dm_1])
    x_interval = (x_max - x_min)/GRID_SIZE
    y_interval = (y_max - y_min)/GRID_SIZE
    interval = max(x_interval,y_interval)
    
    x_list = []
    y_list = []
    for i in range(GRID_SIZE):
        curr_x = np.logical_and(dr_result[:,dm_0] > x_min + interval * i, dr_result[:,dm_0] <= x_min + interval*(i+1) )
        curr_y = np.logical_and(dr_result[:,dm_1] > y_min + interval * i, dr_result[:,dm_1] <= y_min + interval*(i+1) )
        x_list.append(curr_x)
        y_list.append(curr_y)

    fig, ax = plt.subplots()
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            box = np.logical_and(x_list[i], y_list[j])
            
            args_index = np.argwhere(box == True)
            if args_index.size != 0 :
                img_index = args_index[0][0]
                ax.imshow(x_train_org[img_index].reshape(28,28), cmap='gray',\
                            extent=(dr_result[img_index, dm_0], dr_result[img_index, dm_0]+ interval * 0.5,\
                                    dr_result[img_index, dm_1], dr_result[img_index, dm_1]+ interval* 0.5),\
                            zorder=2)
                            
                ax.scatter(dr_result[img_index,dm_0], dr_result[img_index,dm_1], c='r', zorder=1, s = 5)
                
    plt.scatter(dr_result[:,dm_0], dr_result[:,dm_1], s=5, alpha=.5, c='blue', zorder=0)
    
    result_dir = "./result/2/"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    file_name = 'dimension_reduction_scatter.png'   
    plt.axis('equal')
    plt.savefig(result_dir + file_name)
    print("save fig",result_dir + file_name)
    
    if IMG_SHOW:
        plt.show()  
  
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode',
                        default="train",
                        #type=int,
                        dest='MODE',
                        help='train=training and show result, show=show result,(default=train)')
    parser.add_argument('-b',
                        default=64,
                        type=int,
                        dest='BATCH_SIZE',
                        help='batch size,(default=64)')
    parser.add_argument('-lr',
                        default=0.01,
                        type=float,
                        dest='LEARNING_RATE',
                        help='learning rate,(default=0.1)')                        
    parser.add_argument('-ep',
                        default=20,
                        type=int,
                        dest='EPOCH',
                        help='epoch,(default=20)')
    parser.add_argument('-dn',
                        default=1,
                        type=int,
                        dest='DENOISE',
                        help='denoise, 0=False, 1=True,(default=1)')
    parser.add_argument('-drop',
                        default=0.5,
                        type=float,
                        dest='DROPOUT',
                        help='dropout, 0=False, else=dropout_rate,(default=0.5)')                        
    args = parser.parse_args()
    MODE = args.MODE
    BATCH_SIZE = args.BATCH_SIZE
    LEARNING_RATE = args.LEARNING_RATE
    EPOCH = args.EPOCH
    DENOISE = args.DENOISE
    if(args.DROPOUT==0):
        DROPOUT = False
    else:
        DROPOUT = True
    DROPOUT_RATE = args.DROPOUT
    
    print("mode:",MODE)
    print("batch size:",BATCH_SIZE)
    print("learning rate:",LEARNING_RATE)
    print("epoch:",EPOCH)
    print("denoise:",DENOISE)
    print("dropout:",DROPOUT)
    if(DROPOUT):
        print("dropout rate:",DROPOUT_RATE)

    print()
    
    x_train, y_train = loadMNIST(dataset="training", path="MNIST_data")
    x_test, y_test = loadMNIST(dataset="testing", path="MNIST_data")
    
    x_train = x_train/255
    x_test = x_test/255
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)
    
    model_dir = "result/2/"
    model_file_name = "model_2.pickle"
    
    epoch = 20
    train_size = x_train.shape[0]
    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE
    iter_per_epoch = math.ceil(max(train_size / batch_size, 1))
    iters_num = int(iter_per_epoch * epoch)
    denoise = DENOISE
    
    if(MODE == "train"):
        #hw4-2.
        #build network
        train_loss_list = []
        test_loss_list = []
        train_acc_list = []
        test_acc_list = []
        
        network = Modle(input_size=784, learning_rate = learning_rate, lastLayer = SigmoidWithLoss())
        network.add_layer(128,ReLU(),dropout=DROPOUT,dropout_rate=DROPOUT_RATE)
        network.add_layer(784,dropout=False)       
        
        #training
        for i in range(iters_num):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = y_train[batch_mask]
            network.train(x_batch, x_batch,denoise,train_flg=True)            
            if i % iter_per_epoch == 0:
                train_loss = network.loss(x_train, x_train,train_flg=False)
                test_loss = network.loss(x_test, x_test,train_flg=False)
                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)
                if(i == 0):
                    print("loss:")
                else:
                    #loss = sum(network.train_loss_list) / (i+1)
                    train_loss = train_loss_list[-1]
                    test_loss = test_loss_list[-1]
                    print(train_loss)
                    
        #save model
        print("save model...")
        save_model(network, model_dir, model_file_name)

    else:
        #load model
        print("load model...")
        network = load_model(model_dir + model_file_name)
    
    print()
    
    #dimension_reduction
    print("dimension_reduction...")
    batch_mask = np.random.choice(train_size, 500000)
    test_img = x_train[batch_mask]
    dr_result = network.dimension_reduction(test_img,128)
    #save dr_result
    print("save dr_result...")
    np.save("result/2/test_img.npy", test_img)
    np.save("result/2/dr_result.npy", dr_result)
    ##load dr_result
    #print("load dr_result...")
    #test_img = np.load( "result/2/test_img.npy" )
    #dr_result = np.load( "result/2/dr_result.npy" )
    
    #plot dimension reduction scatter
    print("plot dimension reduction scatter:")
    not_zero_idx = np.where(dr_result!=0)
    idx = np.random.choice(not_zero_idx[1], 2, replace=False)
    dm_0 = idx[0]
    dm_1 = idx[1]
    plot_grid_scatter(test_img,dr_result,8,dm_0,dm_1,True)
    
    #Visualize test_img reconstruct comparison
    print("Visualize test_reconstruct comparison:")
    batch_mask = np.random.choice(train_size, 8)
    test_img = x_train[batch_mask]
    test_reconstruct = network.predict2(test_img,train_flg=False)
    draw_compare(test_img, test_reconstruct)
    
    #Visualize filters
    print("Visualize filters:")
    filters = network.layers[0].W
    idx = np.random.choice(128, 16, replace=False)
    filter_list = []
    for f in filters[:, idx].T:
        filter_list.append(f)
    
    visualize_filters(filter_list)
    
    
    