import wandb
import numpy as np
from Neural_Network import * 
from tensorflow.keras.datasets import fashion_mnist

# Define the sweep configuration
sweep_config = {
    'method': 'bayes',  
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'epochs': {'values': [5, 10]},
        'hidden_layers': {'values': [3, 4, 5]},
        'hidden_units': {'values': [32, 64, 128]},
        'weight_decay': {'values': [0, 0.0005, 0.5]},
        'learning_rate': {'values': [1e-3, 1e-4]},
        'optimizer': {'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']},
        'batch_size': {'values': [16, 32, 64]},
        'weight_init': {'values': ['random', 'xavier']},
        'activation': {'values': ['sigmoid', 'tanh', 'relu']}
    }
}

sweep_id = wandb.sweep(sweep_config, project="fashion-mnist-hyperparameter-tuning")

# Load dataset from Fashion MNIST
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalizing the  data
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# splitting for the validation set (10% of training data)
val_size = int(0.1 * X_train.shape[0])
indices = np.random.permutation(X_train.shape[0])  

X_val, y_val = X_train[indices[:val_size]], y_train[indices[:val_size]]
X_train, y_train = X_train[indices[val_size:]], y_train[indices[val_size:]]

y_train_one_hot = one_hot_encode(y_train, 10)
y_val_one_hot = one_hot_encode(y_val, 10)
y_test_one_hot = one_hot_encode(y_test, 10)

# Function to train the model 
def train_model(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Giving a name for this run
        run_name = f"hl_{config.hidden_layers}_bs_{config.batch_size}_ac_{config.activation}_opt_{config.optimizer}"
        wandb.run.name = run_name  

        # Define model with current hyperparameters
        layers = [784] + [config.hidden_units] * config.hidden_layers + [10]
        model = NeuralNetwork(layers)

        # Choosing the  optimizer
        optimizer_dict = {
            'sgd': SGD(config.learning_rate),
            'momentum': Momentum(config.learning_rate, beta=0.9),
            'nesterov': Nesterov(config.learning_rate, beta=0.9),
            'rmsprop': RMSprop(config.learning_rate, beta=0.9),
            'adam': Adam(config.learning_rate),
            'nadam': Nadam(config.learning_rate)
        }
        optimizer = optimizer_dict[config.optimizer]


        best_val_accuracy = 0.0
        val_accuracy_table = wandb.Table(columns=["Epoch", "Validation Accuracy"])

        num_samples = X_train.shape[0]
        for epoch in range(config.epochs):
            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            Y_shuffled = y_train_one_hot[permutation]  

            total_loss, total_accuracy = 0, 0
            num_batches = num_samples // config.batch_size

            for i in range(0, num_samples, config.batch_size):
                X_batch = X_shuffled[i:i + config.batch_size]
                Y_batch = Y_shuffled[i:i + config.batch_size]

                activation_fn = model.forward_propagation(X_batch)
                A_last = activation_fn[f'A{len(layers)-1}']

                loss = model.calculate_loss(Y_batch, A_last)
                grads = model.back_propagation(activation_fn, Y_batch)
                optimizer.step(model.weights, grads)

                predictions = np.argmax(A_last, axis=1)
                labels = np.argmax(Y_batch, axis=1)
                accuracy = np.mean(predictions == labels)

                total_loss += loss
                total_accuracy += accuracy

            avg_loss = total_loss / num_batches
            avg_accuracy = total_accuracy / num_batches

            # Validate the model
            val_activation_fn = model.forward_propagation(X_val)
            val_A_last = val_activation_fn[f'A{len(layers)-1}']
            val_loss = model.calculate_loss(y_val_one_hot, val_A_last)
            val_pred = np.argmax(val_A_last, axis=1)
            val_accuracy = np.mean(np.argmax(y_val_one_hot, axis=1) == val_pred)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

            val_accuracy_table.add_data(epoch + 1, val_accuracy)

            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'train_accuracy': avg_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, step=epoch + 1)

            print(f'Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}, Acc: {avg_accuracy*100:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy*100:.2f}%')
        wandb.log({"Validation Accuracy Scatter": wandb.plot.scatter(val_accuracy_table, "Epoch", "Validation Accuracy", title="Validation Accuracy Scatter")})

      
        wandb.run.summary["best_val_accuracy"] = best_val_accuracy

        # Evaluate the model on the test data
        test_activation_fn = model.forward_propagation(X_test)
        test_A_last = test_activation_fn[f'A{len(layers)-1}']
        test_loss = model.calculate_loss(y_test_one_hot, test_A_last)
        test_pred = np.argmax(test_A_last, axis=1)
        test_accuracy = np.mean(np.argmax(y_test_one_hot, axis=1) == test_pred)

        wandb.run.summary["test_accuracy"] = test_accuracy
        wandb.log({'test_loss': test_loss, 'test_accuracy': test_accuracy})

wandb.agent(sweep_id, train_model, count=10)
