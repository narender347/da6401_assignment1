# da6401_assignment1
For visualizing the  Graphs, Logs other results please refer the WANDB report. link for the report: (https://wandb.ai/narendarhoney-indian-institute-of-technology-madras/fashion-mnist-classification/reports/DA6401-Assignment-1-Submitted-by-NS24Z347-M-NARENDER---VmlldzoxMTY2MTUyNA/edit?draftId=VmlldzoxMTc0MTc0MQ==)
A1_q1.py and A1_q2.py are the codes for the question number 1 and question number 2 respectively.
A1_q1.py program, reads the data from keras.datasets, picks one example from each class and logs the same in the wandb. 
In the A2_q2.py program, The neural network is implemented by the class NeuralNetwork, it initializes the weights randomly and then learns the weights from the training data by forward propagation.
For Question 3, The Batch Size is passed as an integer that determines the size of the mini batch to be taken into consideration. The batch size can be varied in the code Neural_Network.py file in model.fit as shown below
model.fit(X_train, Y_train, optimizer, no_of_epochs=10, batch_size=256, X_val=X_test, Y_val=Y_test)
The Neural_Network.py file consists of forward and backward propogation, the number of layers can be changed from no_of_layers = [784, 128, 64, 10],
784: Input layer (since Fashion MNIST has 28×28 = 784 pixels per image).
128: First hidden layer with 128 neurons.
64: Second hidden layer with 64 neurons.
10: Output layer with 10 neurons (one for each class in Fashion MNIST). the first number indicates the number of inputs and the in between numbers indicate the number of hidden layer inputs and the last number indicates the number of outputs, these can be modified and extra hidden layers can be added in between here, Neural_Network.py file  uses the  optimizer.py file , this optimizer.py file consists of classes related to the sgd, momentum, nesterov, rmsprop, adam, nadam and the optimizer.py file  is also provided.
This Neural_Network.py file is further used for sweeping the hyperparameters and finding out the suitable model to get the good accuracy.
sweep_code.py file is for tuning the hyperparameters and finding out the suitable model to get good accuracy.
The confusion matrix is logged using the code confusion_matrix.py, this program file is used for visualizing the confusion matrix.


