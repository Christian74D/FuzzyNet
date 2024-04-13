#Decision makers for topsis
import torch
from constants import categorize_value

class DecisionMaker:
    def __init__(self, criteria, alternatives):
        self.no_criteria = len(criteria)
        self.no_alternatives = len(alternatives)
        self.criteria = criteria
        self.alternatives = alternatives
        self.reinit()

    def reinit(self):
        self.weights = torch.empty(1, self.no_criteria, 3)
        self.tuple_mat = torch.empty(self.no_alternatives, self.no_criteria, 3)
        self.tuple_weights = torch.empty(1, self.no_criteria, 3)
        self.matrix = torch.empty(self.no_alternatives, self.no_criteria, dtype=torch.long)


    def set_ip(self):
        for i in range(self.no_alternatives):
            for j in range(self.no_criteria):
                entry = self.matrix[j][i].unsqueeze(0)
                new_entry = torch.cat((entry + 1, entry + 2, entry + 3))
                self.tuple_mat[i][j] = new_entry


        self.load_weights()

    def load_weights(self):
      for k in range(self.no_criteria):
            self.tuple_weights[0][k] = torch.tensor((k+1, k +2, k + 3))

    def load_comb_weights(self):
      for k in range(self.no_criteria):
            m = torch.round(self.weights[k] * 10).unsqueeze(0)
            m = torch.cat((m, m+1, m+2), 0)
            self.tuple_weights[0][k] = m

    #display decimal values
    def display_state_val(self):
        print(self.matrix.t())

    def display_weights_val(self):
        print(self.weights)

    #display linguistic variables
    
    def display_state(self):
        rows = []
        for row in self.matrix.t():
            row_categories = [categorize_value(value.item()) for value in row]
            rows.append(row_categories)

        # Print table
        for row in rows:
            print("|", end="")
            for category in row:
                print(f" {category} |", end="")
            print()


    def display_weights(self):
        categories = [categorize_value(value.item()*5) for value in self.weights]
        print("|", end="")
        for category in categories:
            print(f" {category} |", end="")
        print()
        print()
