import argparse
import numpy as np
import wandb
from Neural_Network import *

run = wandb.init(project="DA6401_Assignment1",name='code_specs')

parser = argparse.ArgumentParser(description='Neural Network Code Specifications')

parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname', help='Project name used to track experiments in Weights & Biases dashboard')
parser.add_argument('-we', '--wandb_entity', type=str, default='myname', help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
parser.add_argument('-sid', '--wandb_sweepid', type=str, default=None, help='Wandb Sweep Id to log in sweep runs the Weights & Biases dashboard.')
parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', choices=["mnist", "fashion_mnist"], help='Dataset choices: ["mnist", "fashion_mnist"]')
parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of epochs to train neural network.')
parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size used to train neural network.')
parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=["MSE", "cross_entropy"], help='Loss function choices: ["MSE", "cross_entropy"]')
parser.add_argument('-o', '--optimizer', type=str, default='adam', choices=["sgd", "momentum", "nesterov", "rmsprop", "adam"], help='Optimizer choices: ["sgd", "momentum", "nesterov", "rmsprop", "adam"]')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help='Learning rate used to optimize model parameters')
parser.add_argument('-m', '--momentum', type=float, default=0.9, help='Momentum used by momentum and nag optimizers.')
parser.add_argument('-beta', '--beta', type=float, default=0.99, help='Beta used by rmsprop optimizer')
parser.add_argument('-beta1', '--beta1', type=float, default=0.9, help='Beta1 used by adam and nadam optimizers.')
parser.add_argument('-beta2', '--beta2', type=float, default=0.999, help='Beta2 used by adam and nadam optimizers.')
parser.add_argument('-eps', '--epsilon', type=float, default=1e-08, help='Epsilon used by optimizers.')
parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0005, help='Weight decay used by optimizers.')
parser.add_argument('-w_i', '--weight_init', type=str, default='xavier', choices=["random_init", "xavier_init"], help='Weight initialization choices: ["random_init", "xavier_init"]')
parser.add_argument('-nhl', '--num_layers', type=int, default=5, help='Number of hidden layers used in feedforward neural network.')
parser.add_argument('-sz', '--hidden_size', type=int, default=128, help='Number of hidden neurons in a feedforward layer.')
parser.add_argument('-a', '--activation_func', type=str, default='tanh', choices=["sigmoid", "tanh", "relu"], help='Activation function choices: ["sigmoid", "tanh", "relu"]')


args = parser.parse_args()

#train_data, test_data = None, None
if args.dataset == 'mnist':
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
elif args.dataset == 'fashion_mnist':
    from keras.datasets import fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

input_size = 28 * 28  # Flattened image size
output_size = 10  # Number of classes

# Load the Fashion MNIST dataset
#(X_train, y_train), (X_test, y_test) = train_data

# normalizes the data
X_train = X_train.reshape(-1, 28*28) / 255.0  # Flatten the images and normalize
X_test = X_test.reshape(-1, 28*28) / 255.0

# One-hot encode the labels
y_train_one_hot = np.zeros((y_train.shape[0], 10))
y_train_one_hot[np.arange(y_train.shape[0]), y_train] = 1

y_test_one_hot = np.zeros((y_test.shape[0], 10))
y_test_one_hot[np.arange(y_test.shape[0]), y_test] = 1

if not args.wandb_sweepid is None:
    # For sweep runs, where config is set by wandb.ai
    def train(config=None):
    # Initialize a new wandb run
        with wandb.init(config=config) as run:

            config = wandb.config
            
            run_name = str(config).replace("': '", ' ').replace("'", '')
            print(run_name)
            run.name = run_name
            hidden_layers=[]
            no_of_hidden_layer=config.hidden_layers
            hidden_layer_size=config.hidden_layer_size
            for i in range(no_of_hidden_layer):
                 hidden_layers.append(hidden_layer_size)   # eg :-  hidden_layers = [128, 128] , Number of neurons in each hidden layer
    
            nn = NeuralNetwork(input_size, hidden_layers, output_size,
                               activation=config.activation_func,weight_init=config.weight_init,loss_func='cross_entropy', 
                               learn_rate=config.learning_rate, grad_desc=config.optimizer)
            
            nn.train(X_train, y_train_one_hot, batch_size=config.batch_size, epochs=config.epochs,val_split=0.1)
    wandb.agent(args.wandb_sweepid,entity=args.wandb_entity, project=args.wandb_project, function=train)

else:
    hidden_layers=[]
    no_of_hidden_layer=args.num_layers
    hidden_layer_size=args.hidden_size
    for i in range(no_of_hidden_layer):
        hidden_layers.append(hidden_layer_size)

    nn = NeuralNetwork(input_size, hidden_layers, output_size,
                               activation=args.activation_func,weight_init=args.weight_init,loss_func=args.loss, 
                               learn_rate=args.learning_rate, grad_desc=args.optimizer,beta_1=args.beta1,beta_2=args.beta2,epsilon=args.epsilon,
                               rho=args.beta,momentum=args.momentum)
    nn.train(X_train, y_train_one_hot, batch_size=args.batch_size, epochs=args.epochs,val_split=0.1)
        
