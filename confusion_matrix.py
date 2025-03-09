import wandb
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from Neural_Network import *


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

y_train_one_hot = one_hot_encode(y_train, 10)
y_test_one_hot = one_hot_encode(y_test, 10)

# Defining the sweep configuration 
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

sweep_id = wandb.sweep(sweep_config, project="fashion-mnist-sweep")

# Function to train the model 
def train_model(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
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

        # Train the  model
        model.fit(X_train, y_train_one_hot, optimizer, config.epochs, config.batch_size, X_val=X_test, Y_val=y_test_one_hot)

        # Evaluate model on validation set
        val_activation_fn = model.forward_propagation(X_test)
        val_A_last = val_activation_fn[f'A{len(layers)-1}']
        val_accuracy = np.mean(np.argmax(y_test_one_hot, axis=1) == np.argmax(val_A_last, axis=1))

       #logging in wandb
        wandb.run.summary["val_accuracy"] = val_accuracy
        wandb.log({"val_accuracy": val_accuracy})

wandb.agent(sweep_id, train_model, count=10)


api = wandb.Api()
sweep = api.sweep(f"fashion-mnist-sweep/{sweep_id}")

best_run = sorted(sweep.runs, key=lambda run: run.summary.get("val_accuracy", 0), reverse=True)[0]

# Extract best model's hyperparameters
best_config = best_run.config
print(f"Best model found: {best_config}")

# Initialize WandB for logging best model evaluation
wandb.init(project="fashion-mnist-best-model", name="best_model_confusion_matrix")

# Define and initialize the best model
best_layers = [784] + [best_config['hidden_units']] * best_config['hidden_layers'] + [10]
best_model = NeuralNetwork(best_layers)

# Choose thee optimizer
best_optimizer = {
    'sgd': SGD(best_config['learning_rate']),
    'momentum': Momentum(best_config['learning_rate'], beta=0.9),
    'nesterov': Nesterov(best_config['learning_rate'], beta=0.9),
    'rmsprop': RMSprop(best_config['learning_rate'], beta=0.9),
    'adam': Adam(best_config['learning_rate']),
    'nadam': Nadam(best_config['learning_rate'])
}[best_config['optimizer']]

# Train the best model again on full training data
best_model.fit(X_train, y_train_one_hot, best_optimizer, best_config['epochs'], best_config['batch_size'])

# Now Evaluate the best model on test data
test_activation_fn = best_model.forward_propagation(X_test)
test_A_last = test_activation_fn[f'A{len(best_layers)-1}']
test_pred = np.argmax(test_A_last, axis=1)

# Computing the  confusion matrix 
num_classes = 10
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

for true_label, pred_label in zip(y_test, test_pred):
    conf_matrix[true_label, pred_label] += 1


test_loss = best_model.calculate_loss(y_test_one_hot, test_A_last)
test_accuracy = np.mean(y_test == test_pred)
wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})


plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, cmap="Blues")
plt.colorbar()


for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, str(conf_matrix[i, j]), ha="center", va="center", color="black")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Best Model")
plt.xticks(range(num_classes), name_of_classes, rotation=45)
plt.yticks(range(num_classes), name_of_classes)

wandb.log({"Confusion Matrix": wandb.Image(plt)})
plt.close()

print(f"Best Model Test Loss: {test_loss:.4f}")
print(f"Best Model Test Accuracy: {test_accuracy * 100:.2f}%")
