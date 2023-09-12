from abc import ABC, abstractmethod
import numpy as np
np.random.seed(0)

class Layer(ABC):
    def __init__(self):
        self.prevIn = []
        self.prevOut = []
    def setPrevIn(self, dataIn):
        self.prevIn = dataIn

    def setPrevOut(self, out):
        self.prevOut = out

    def getPrevIn(self):
        return self.prevIn

    def getPrevOut(self):
        return self.prevOut

    @abstractmethod
    def forward(self, dataIn):
        pass

    @abstractmethod
    def gradient(self):
        pass

    @abstractmethod
    def backward(self, gradIn):
        pass

class InputLayer(Layer):
    def __init__(self, dataIn):
        super().__init__()
        self.meanX = np.mean(dataIn, axis=0, keepdims=True)
        self.stdX = np.std(dataIn, axis=0, ddof=1, keepdims=True)
        self.stdX[self.stdX == 0] = 1

    def forward(self, dataIn, score = False):
        if score == True:
            y = (dataIn - self.meanX) / self.stdX
            self.setPrevIn(dataIn)
            self.setPrevOut(y)
            return y
        else:
            self.setPrevOut(dataIn)
            return dataIn
    def gradient(self):
        pass


    def backward(self, gradIn):
        pass


class LinearLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        y = dataIn
        self.setPrevOut(y)
        return y

    def gradient(self):
        return np.ones_like(self.getPrevIn())

    def backward(self, gradIn):
        return gradIn * self.gradient()

class ReLuLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        y = np.maximum(0, dataIn)
        self.setPrevOut(y)
        return y

    def gradient(self):
        return np.where(self.getPrevIn() > 0, np.ones_like(self.getPrevIn()), 0)


    def backward(self, gradIn):
        return gradIn * self.gradient()


class LogisticSigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        y = 1 / (1 + np.exp(-dataIn))
        self.setPrevOut(y)
        return y

    def gradient(self):
        g = self.getPrevOut()
        return g * (1 - g)

    def backward(self, gradIn):
        return gradIn * self.gradient()

class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        
        shiftx = dataIn - np.max(dataIn, axis=-1, keepdims=True)
        
        exp_data = np.exp(shiftx)
        sum_exp_data = np.sum(exp_data, axis=-1, keepdims=True)
        y = exp_data / sum_exp_data
        self.setPrevOut(y)
        return y
    def gradient(self):
        g = self.getPrevOut()
        batch_size, num_classes = g.shape
        grad = np.zeros((batch_size, num_classes, num_classes))

        for i in range(batch_size):
            grad[i] = np.diagflat(g[i])-np.outer(g[i], g[i])

        return grad 

    def backward(self, gradIn):
        grad = self.gradient()
        return np.einsum('bij,bj->bi', grad, gradIn)
    

class TanhLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        y = np.tanh(dataIn)
        self.setPrevOut(y)  
        return y

    def gradient(self):
        g = self.getPrevOut()
        return 1 - g**2

    def backward(self, gradIn):
        return gradIn * self.gradient()
    

class FullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut):
        super().__init__()
        self.weights = np.random.randn(sizeIn, sizeOut) * 1e-4
        self.biases = np.random.randn(1, sizeOut) * 1e-4
        self.__sw = 0
        self.__sb = 0
        self.__rw = 0
        self.__rb = 0
        
    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        self.weights = weights

    def getBiases(self):
        return self.biases

    def setBiases(self, biases):
        self.biases = biases

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        y = np.dot(dataIn, self.weights) + self.biases
        self.setPrevOut(y)
        return y

    def gradient(self):
        return self.getWeights().T

    def backward(self, gradIn):        
        return gradIn @ self.gradient()
    
    def updateWeights(self,gradIn,eta = 0.0001,t=1,p1=1,p2=1,delta=1,adam=False):
        #print('gradIn for updateWeights: \n', gradIn)
        dJdb = np.sum(gradIn, axis = 0)/gradIn.shape[0]
        #print('dJdb: \n', dJdb)
        dJdW = (self.getPrevIn().T @ gradIn)/gradIn.shape[0]
        #print('dJdW: \n', dJdW)
        
        #print('previous weights: \ n',self.getWeights())
        #print('previous biases: \n',self.getBiases())
        
        #print('eta*dJdW: \n',eta*dJdW)
        #print('eta*dJdb: \n',eta*dJdb)
        #print()
        
        #added the ability to add in adam or not
        
        if adam == True:
            self.__sw = p1*self.__sw + (1-p1)*dJdW
            self.__rw = p2*self.__rw + (1-p2)*(dJdW*dJdW)
            
            self.__sb = p1*self.__sb + (1-p1)*dJdb
            self.__rb = p2*self.__rb + (1-p2)*(dJdb*dJdb)
            
            wadam = (self.__sw/(1-p1**t))/(np.sqrt(self.__rw/(1-p2**t)) + delta)
            badam = (self.__sb/(1-p1**t))/(np.sqrt(self.__rb/(1-p2**t)) + delta)
            
            self.weights -= eta*wadam
            self.biases -= eta*badam
            
        else:
            self.weights -= eta*dJdW
            self.biases -= eta*dJdb
    

class SquaredError:
    def eval(self, Y, Yhat):
        return np.mean((Y - Yhat)*(Y - Yhat))

    def gradient(self, Y, Yhat):
        return -2 * (Y-Yhat)


class LogLoss:
    def eval(self, Y, Yhat):
        return np.mean(-(Y*np.log(Yhat+0.0000001)+(1-Y)*np.log(1-Yhat+0.0000001)))

    def gradient(self, Y, Yhat):
        return -(Y-Yhat) / (Yhat * (1-Yhat)+0.0000001)

class CrossEntropy:
    def eval(self, Y, Yhat):
         return -np.mean(np.sum(Y * np.log(Yhat+0.0000001),axis=1))

    def gradient(self, Y, Yhat):
        return -(Y / (Yhat+0.0000001))


class DropoutLayer:
    def __init__(self, p=0.5):
        self.p = p  
        self.train_mode = True
        self.mask = None

    def forward(self, inputs):
        if self.train_mode:

            self.mask = np.random.binomial(1, 1-self.p, size=inputs.shape) 
            return inputs * self.mask / (1-self.p)
        else:
            return inputs

    def backward(self, dinputs):
        if self.train_mode:
            return dinputs * self.mask
        else:
            return dinputs

    def updateWeights(self, *args):
        pass
    

class ConvolutionalLayer(Layer):
    def __init__(self, input_channels, num_filters, kernel_size, stride=1, padding='valid'):
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.filters = np.random.randn(num_filters, input_channels, kernel_size, kernel_size) * np.sqrt(2. / (kernel_size*kernel_size))
        self.biases = np.zeros(num_filters)
        
        self.__sw = np.zeros_like(self.filters)
        self.__rw = np.zeros_like(self.filters)
        self.__sb = np.zeros_like(self.biases)
        self.__rb = np.zeros_like(self.biases)
        
        self.d_filters = None
        self.d_biases = None
        self.prev_input = None

    def forward(self, dataIn):
        self.setPrevIn(dataIn)

        self.prev_input = dataIn.copy()
        batch_size, in_channels, height, width = dataIn.shape
        
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        
        output = np.zeros((batch_size, self.num_filters, out_height, out_width))

        for i in range(0, out_height):
            for j in range(0, out_width):
                slice = dataIn[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                for f in range(self.num_filters):
                    output[:, f, i, j] = np.sum(slice * self.filters[f], axis=(1, 2, 3)) + self.biases[f]

        self.setPrevOut(output)
        return output

    def backward(self, gradIn):
        
        batch_size, _, out_height, out_width = gradIn.shape
        d_input = np.zeros_like(self.prev_input)
        d_filters = np.zeros_like(self.filters)
        d_biases = np.sum(gradIn, axis=(0, 2, 3))

        for i in range(0, out_height * self.stride, self.stride):
            for j in range(0, out_width * self.stride, self.stride):
                slice = self.prev_input[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                for f in range(self.num_filters):
                    d_input[:, :, i:i+self.kernel_size, j:j+self.kernel_size] += gradIn[:, f, i//self.stride, j//self.stride][:, None, None, None] * self.filters[f]
                    d_filters[f] += np.sum(slice * gradIn[:, f, i//self.stride, j//self.stride][:, None, None, None], axis=0)

        self.d_filters = d_filters
        self.d_biases = d_biases
        return d_input

    def gradient(self):
        return self.d_filters, self.d_biases

    def updateWeights(self, grad, eta=0.0001, t=1, p1=0.9, p2=0.999, delta=1e-7, adam=False):
        if adam:
            self.__sw = p1 * self.__sw + (1 - p1) * self.d_filters
            self.__rw = p2 * self.__rw + (1 - p2) * (self.d_filters * self.d_filters)

            self.__sb = p1 * self.__sb + (1 - p1) * self.d_biases
            self.__rb = p2 * self.__rb + (1 - p2) * (self.d_biases * self.d_biases)

            wadam = (self.__sw / (1 - p1**t)) / (np.sqrt(self.__rw / (1 - p2**t)) + delta)
            badam = (self.__sb / (1 - p1**t)) / (np.sqrt(self.__rb / (1 - p2**t)) + delta)

            self.filters -= eta * wadam
            self.biases -= eta * badam
        else:
            self.filters -= eta * self.d_filters
            self.biases -= eta * self.d_biases
            
    def setWeights(self, weights):
        self.filters = weights

class MaxPoolingLayer(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = {}  
        self.max_indices = None

    def forward(self, input_tensor):
        self.setPrevIn(input_tensor) 
        N, C, H, W = input_tensor.shape
        HO = (H - self.pool_size) // self.stride + 1
        WO = (W - self.pool_size) // self.stride + 1
        
        output = np.zeros((N, C, HO, WO))
        
        for h in range(HO):
            for w in range(WO):
                h_start, h_end = h * self.stride, h * self.stride + self.pool_size
                w_start, w_end = w * self.stride, w * self.stride + self.pool_size
                
                patch = input_tensor[:, :, h_start:h_end, w_start:w_end]
                output[:, :, h, w] = np.max(patch, axis=(2, 3))
                
                max_indices = np.argmax(patch.reshape(N, C, -1), axis=2)
                for n in range(N):
                    for c in range(C):
                        self.cache[(n, c, h, w)] = np.unravel_index(max_indices[n, c], (self.pool_size, self.pool_size))
                    
        return output

    def backward(self, d_output):
        N, C, HO, WO = d_output.shape
        d_input = np.zeros_like(self.getPrevIn()) 
            
        for n in range(N):
            for c in range(C):
                for h in range(HO):
                    for w in range(WO):
                        (h_max, w_max) = self.cache[(n, c, h, w)]
                        h_start = h * self.stride
                        w_start = w * self.stride
                            
                        d_input[n, c, h_start + h_max, w_start + w_max] = d_output[n, c, h, w]
                            
            return d_input

    def gradient(self):
        pass
    
    
class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self.setPrevIn(input_data)
        batch_size = input_data.shape[0]
        output_data = input_data.reshape(batch_size, -1)
        self.setPrevOut(output_data)
        return output_data

    def backward(self, gradIn):

        return gradIn.reshape(self.getPrevIn().shape)

    def gradient(self):
        pass