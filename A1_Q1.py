from tensorflow import keras
import wandb
import numpy as np
import matplotlib.pyplot as plt


wandb.login()
# Initialize wandb with project and name 
wandb.init(project="fashion-mnist-classification", name ="visualising_the_input" )

# Load Fashion MNIST datase
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Define class labels
label_of_classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Select one sample for each class
eg_image = []
label_eg = []
for label in range(10):
    index_eg = np.where(y_train == label)[0][0]  # Get first occurrence of each class
    eg_image.append(x_train[index_eg])
    label_eg.append(label_of_classes[label])

# Plot and log images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(eg_image[i], cmap='gray')
    ax.set_title(label_eg[i])
    ax.axis('off')

plt.tight_layout()
wandb.log({"Samples from the mnist fashion for classification": wandb.Image(fig)})
plt.close(fig)

# Finish wandb run
wandb.finish()
