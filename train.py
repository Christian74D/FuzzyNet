import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from constants import num_epochs, lr
from model import FuzzyNet
from data import train_loader
from plots import plot_loss

# Step 4: Define Loss Function and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu" #probably cpu is faster since we are using only one input at a time


print(f"Initializing device <{device}>")
model = FuzzyNet(device)
model.to(device)

loss_list = []
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


def main():
    print("Initializing training loop...")
    print(f"Training for {num_epochs} epochs")
    # Step 5: Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
            optimizer.zero_grad()
            _, outputs = model(inputs.squeeze())
            outputs = outputs.unsqueeze(0).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        loss_list.append(loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    model.save()
    plot_loss(loss_list)

if __name__ == "__main__":
    main()