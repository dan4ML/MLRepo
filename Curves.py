import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a Pandas DataFrame
csv_file = "path/to/metricfile.csv" 

data = pd.read_csv(csv_file)

# Extract columns
epochs = data["Epoch"]
validation_loss = data["Validation Loss"]
training_loss = data["Training Loss"]
validation_accuracy = data["Validation Accuracy"]
training_accuracy = data["Training Accuracy"]

# Plot 1: Epoch vs Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, validation_loss, label="Validation Loss", marker='o', color='blue')
plt.plot(epochs, training_loss, label="Training Loss", marker='o', color='orange')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Epoch vs Loss")
plt.legend()
plt.grid()
plt.show()

# Plot 2: Epoch vs Accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs, validation_accuracy, label="Validation Accuracy", marker='o', color='green')
plt.plot(epochs, training_accuracy, label="Training Accuracy", marker='o', color='red')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Epoch vs Accuracy")
plt.legend()
plt.grid()
plt.show()
