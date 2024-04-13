#Fuzzy topsis implementation
import torch
from decision_maker import DecisionMaker

#distance bet 2 fuzzy value sets
def fuzzy_dist(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2))

    
#topsis model
class FuzzyTopsis:
    def __init__(self, criteria, alternatives, no_dms):
        self.no_criteria = len(criteria)
        self.no_alternatives = len(alternatives)
        self.criteria = criteria
        self.alternatives = alternatives
        self.no_dms = no_dms
        self.combined_dm = None
        self.dms = [DecisionMaker(criteria, alternatives) for _ in range(no_dms)]
        self.fpis = self.fnis = None
        self.inputs = None
        self.display = False


    def calc_fpis_fnis_mats(self):
        self.fpis = torch.empty(self.no_alternatives, self.no_criteria)
        self.fnis = torch.empty(self.no_alternatives, self.no_criteria)
        for i, col in enumerate(self.combined_dm.tuple_mat):
            Ap = col[0]
            An = col[0]
            for entry in col:
                if torch.gt(entry, Ap).any():
                    Ap = entry
                if torch.lt(entry, An).any():
                    An = entry
            fp = fuzzy_dist(Ap.unsqueeze(0), col)
            fn = fuzzy_dist(An.unsqueeze(0), col)

            self.fpis[i] = fp
            self.fnis[i] = fn


    def display_fnis_fpis(self):
        print("fpis weighted mat")
        print(self.fpis)
        print("fnis weighted mat")
        print(self.fnis)

    def display_state_mat(self):
        for i, dm in enumerate(self.dms):
            print("Decision Maker ", i)
            dm.display_state()

    def display_weight_comb(self):
        print("Decision Makers combined weights")
        self.combined_dm.display_weights()

    def apply_weights(self):
        temp = torch.cat((self.combined_dm.tuple_weights,self.combined_dm.tuple_weights,self.combined_dm.tuple_weights,self.combined_dm.tuple_weights), 0)
        self.combined_dm.tuple_mat = torch.mul(temp, self.combined_dm.tuple_mat)

    def normalize(self):
        min_set = torch.max(self.combined_dm.tuple_mat[:, :, 2], dim=0)[0].unsqueeze(0).unsqueeze(2)
        self.combined_dm.tuple_mat = torch.div(self.combined_dm.tuple_mat, min_set)



    def calc_cci(self):
        dip = torch.sum(self.fpis, dim=1)
        din = torch.sum(self.fnis, dim=1)
        self.cci = din / (din + dip)

    def aggregate(self):
        for i in range(self.no_alternatives):
            for j in range(self.no_criteria):
                min_vals = [dm.tuple_mat[i][j][0] for dm in self.dms]
                self.combined_dm.tuple_mat[i][j][0] = torch.min(torch.stack(min_vals))

                avg_vals = [dm.tuple_mat[i][j][1] for dm in self.dms]
                self.combined_dm.tuple_mat[i][j][1] = torch.mean(torch.stack(avg_vals))

                max_vals = [dm.tuple_mat[i][j][2] for dm in self.dms]
                self.combined_dm.tuple_mat[i][j][2] = torch.max(torch.stack(max_vals))


    def eval(self):
        self.aggregate()
        if self.display:
          print()
          self.display_state_mat()
          print()
          self.display_weight_comb()
          print()

        self.normalize()
        self.apply_weights()
        self.calc_fpis_fnis_mats()
        self.calc_cci()
        max_index = torch.argmax(self.cci)
        return max_index, self.cci[max_index], self.cci


    def update_matrices(self, model_output, weights):
        self.inputs = model_output
        self.combined_dm = DecisionMaker(self.criteria, self.alternatives)
        for i, matrix_output in enumerate(model_output):
            dm = self.dms[i]
            dm.reinit()
            dm.matrix = matrix_output
            dm.set_ip()


        self.combined_dm.weights = weights
        self.combined_dm.load_comb_weights()

