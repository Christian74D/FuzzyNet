import torch
import torch.nn as nn
from fuzzy_topsis import FuzzyTopsis
from data import column_names, no_classes
from constants import fuzzy_size, model_path, no_dms

class FuzzyNet(nn.Module):
    def __init__(self, device, num_networks=6, hidden_size=6):
        super().__init__()
        self.num_networks = num_networks
        self.hidden_size = hidden_size

        self.networks = nn.ModuleList([nn.ModuleList([self.create_network().to(device) for _ in range(num_networks)]) for i in range(no_dms)])
        self.ft = FuzzyTopsis(column_names, range(no_classes), no_dms)
        self.weights = nn.Linear(1, 6)
        self.device = device

    def display(self, val):
      self.ft.display = val

    def create_network(self):
        return nn.Sequential(
            nn.Linear(1, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, no_classes)
        )

    def forward(self, x):
      op = torch.tensor([]).to(self.device)
      for dm_net in (self.networks):
          y = torch.tensor([]).to(self.device)
          for i, network in enumerate(dm_net):
              z = x[i].unsqueeze(0)
              outputs = network(z)

              predicted_classes = torch.sigmoid(outputs) * fuzzy_size
              y = torch.cat((y, predicted_classes.unsqueeze(0)),0)
          op = torch.cat((op, y.unsqueeze(0)), 0)


      #dummy tensor for weights
      dummy = torch.tensor([1.0], requires_grad = True).to(self.device)
      weights = (torch.sigmoid(self.weights(dummy)) + 1) / 2

      self.ft.update_matrices(op, weights)

      max_index, cci_max, cci_list = self.ft.eval()
      return max_index, cci_list


    def save(self):
        

        print(f"Model saved at {model_path}")
        torch.save(self.state_dict(), model_path)


def load_model(device):
    model = FuzzyNet(device).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

