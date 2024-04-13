import matplotlib.pyplot as plt
from constants import plot_loss_path
def plot_loss(loss_list, save_path=None):
    epochs = range(1, len(loss_list) + 1)

    # Plot loss vs epochs
    plt.plot(epochs, loss_list, marker='o', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.grid(True)

    plt.savefig(plot_loss_path)  
    print(f"Plot saved as: {plot_loss_path}")

    #plt.show()  #

