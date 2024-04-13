from model import load_model
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from data import test_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading saved model..")
model = load_model(device)

model.to(device)

def main():

    # Initialize lists to store true labels and predicted labels
    true_labels = []
    pred_labels = []

    # Set the model to evaluation mode
    model.eval()
    model.display(True)
    total = 0
    correct = 0

    print('First input\'s decision making displayed for example')

    # Iterate through the test DataLoader to get predictions
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
            _, outputs = model(inputs.squeeze())
            outputs = outputs.unsqueeze(0).to(device)

            _, predicted = torch.max(outputs.data, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            model.display(False)

    # Convert lists to numpy arrays
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    print("Confusion Matrix:")
    print(conf_matrix)

    # Compute accuracy
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")



if __name__ == "__main__":
    main()