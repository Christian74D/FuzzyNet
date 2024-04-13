import os

#dir
current_dir = str(os.path.dirname(__file__)) + '\\'

#fuzzy variables (VL L M H VH)
fuzzy_size = 5  
no_dms = 3


#train-test ratio
test_size = 0.2

# num of train epochs
num_epochs = 5
lr = 0.0025

#model path(save/load)
plot_loss_path = current_dir + "train_losses.jpg"
model_path = current_dir + "models\FuzzyNet.pth"
inference_img_path = current_dir + "inference.jpg"


#fuzzy category values
def categorize_value(value):
    if value < 1:
        return 'VL'
    elif value < 2:
        return 'L'
    elif value < 3:
        return 'M'
    elif value < 4:
        return 'H'
    else:
        return 'VH'
