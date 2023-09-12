from layers import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
#import os
#import sys

X_train = pd.read_csv('trainingfull.csv',header=None).values
X_valid = pd.read_csv('testingfull.csv',header=None).values
Y_train = pd.read_csv('trainingfull_labels.csv',header=None).values
Y_valid = pd.read_csv('testingfull_labels.csv',header=None).values

X_train = X_train.reshape(-1, 128, 128, 1)
X_valid = X_valid.reshape(-1, 128, 128, 1)
X_train = X_train.transpose(0, 3, 1, 2)
X_valid = X_valid.transpose(0, 3, 1, 2)

Y_train = np.array([Y_train])
Y_train = Y_train.flatten().astype(int)

Y_valid = np.array([Y_valid])
Y_valid = Y_valid.flatten().astype(int)


def one_hot_encode(labels, num_classes=4):
    one_hot_encoded = np.eye(num_classes)[labels]
    return one_hot_encoded

X_train = X_train / 255.0
X_valid = X_valid / 255.0

y_train_raw = Y_train
y_valid_raw = Y_valid
y_train = one_hot_encode(y_train_raw)
y_valid = one_hot_encode(y_valid_raw)


filter_size = 3
pool_size , stride = 2 , 2
num_filters = 1

dim_flat = (X_train.shape[0], 63 * 63 * num_filters)

weights = np.random.randn(num_filters, 1, filter_size, filter_size) * np.sqrt(2. / (filter_size * filter_size))

input_layer = InputLayer(X_train)
conv_layer = ConvolutionalLayer(input_channels=1, 
                                num_filters = num_filters,
                                kernel_size = filter_size, 
                                stride=1)
conv_layer.setWeights(weights)
relu_layer1 = ReLuLayer()
maxpool = MaxPoolingLayer(pool_size= pool_size, stride=stride)
flatlayer = FlattenLayer()

fc_layer1 = FullyConnectedLayer(dim_flat[1], 4)
fc_layer1.setWeights(np.random.randn(dim_flat[1],4) * np.sqrt(2. / dim_flat[1]))
fc_layer1.setBiases(np.zeros((1, 4)))

softmax_layer = SoftmaxLayer()
objective = CrossEntropy()
dropout1 = DropoutLayer(0.5)
#dropout2 = DropoutLayer(0.2)


layers = [input_layer, conv_layer, relu_layer1, dropout1, maxpool , flatlayer,  fc_layer1, softmax_layer]

eta = 0.001
epochs = 350

losses = []
validation_losses = []

for epoch in range(epochs):
    if (epoch+1) % 5 == 0:
        print(f"Epoch: {epoch+1}")
    dropout1.train_mode = True
    #dropout2.train_mode = True

    h = X_train
    for layer in layers:
       #print(f"Layer {layer.__class__.__name__}: Input shape: {h.shape}")

        h = layer.forward(h)
        #print(h.shape)
    loss = objective.eval(y_train, h)
    losses.append(loss)

    grad = objective.gradient(y_train, h)
    
    #print(grad.shape)
    for layer in reversed(layers):

        newgrad = layer.backward(grad)
        if isinstance(layer, ConvolutionalLayer):
            layer.updateWeights(grad, eta)
        if isinstance(layer, FullyConnectedLayer):
            dJdb = np.sum(grad, axis=0) / grad.shape[0]
            dJdW = (layer.getPrevIn().T @ grad) / grad.shape[0]

            #dJdW += 0.001 * layer.weights
            
            layer.weights -= eta * dJdW
            layer.biases -= eta * dJdb
       #print(f"end {layer.__class__.__name__}: Gradient Input shape: {grad.shape} , Gradient Output shape: {newgrad.shape}")
    
        grad=newgrad


    dropout1.train_mode = False
    #dropout2.train_mode = False
    h_valid = X_valid
    for layer in layers:
        h_valid = layer.forward(h_valid)

    val_loss = objective.eval(y_valid, h_valid)
    validation_losses.append(val_loss)

plt.plot(losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

y_train_pred = np.argmax(h, axis=1)
y_valid_pred = np.argmax(h_valid, axis=1)

train_accuracy = np.mean(y_train_pred == y_train_raw)
valid_accuracy = np.mean(y_valid_pred == y_valid_raw)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {valid_accuracy * 100:.2f}%")
def confusion_matrix(true_labels, predicted_labels):
    classes = sorted(np.unique(true_labels))
    matrix = np.zeros((len(classes), len(classes)))

    for t, p in zip(true_labels, predicted_labels):
        matrix[classes.index(t), classes.index(p)] += 1
        
    return matrix

def classification_report(true_labels, predicted_labels):
    matrix = confusion_matrix(true_labels, predicted_labels)
    
    precision = np.diag(matrix) / np.sum(matrix, axis=0)
    recall = np.diag(matrix) / np.sum(matrix, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("Class\tPrecision\tRecall\tF1-Score")
    for i, label in enumerate(sorted(np.unique(true_labels))):
        print(f"{label}\t{precision[i]:.2f}\t\t{recall[i]:.2f}\t{f1_score[i]:.2f}")

    print("\nOverall Precision: {:.2f}".format(np.mean(precision)))
    print("Overall Recall: {:.2f}".format(np.mean(recall)))
    print("Overall F1-Score: {:.2f}".format(np.mean(f1_score)))
print("Confusion Matrix:")
print(confusion_matrix(y_valid_raw, y_valid_pred))
print("\nClassification Report:")
classification_report(y_valid_raw, y_valid_pred)