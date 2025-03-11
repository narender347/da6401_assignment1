# da6401_assignment1
The Neural_Network.py file consists of forward and backward propogation, the number of layers can be changed from no_of_layers = [784, 128, 64, 10],
784: Input layer (since Fashion MNIST has 28×28 = 784 pixels per image).
128: First hidden layer with 128 neurons.
64: Second hidden layer with 64 neurons.
10: Output layer with 10 neurons (one for each class in Fashion MNIST). the first number indicates the number of inputs and the in between numbers indicate the number of hidden layer inputs and the last number indicates the number of outputs, these can be modified and extra hidden layers can be added in between here, Neural_Network.py file  uses the  optimizer.py file , this optimizer.py file consists of classes related to the sgd, momentum, nesterov, rmsprop, adam, nadam and the optimizer.py file  is also provided.
This Neural_Network.py file is further used for sweeping the hyperparameters and finding out the suitable model to get the good accuracy.
sweep_code.py file is for tuning the hyperparameters and finding out the suitable model to get good accuracy.
confusion_matrix.py is for creating the confusion matrix.
A1_Q1 and A1_Q2 are the codes for the question number 1 and question number 2 respectively.
