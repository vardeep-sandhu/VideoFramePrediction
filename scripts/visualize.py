import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import utils 
from model import Model
from dataset import MNIST_Moving


def show(grid, name):
    fix, axs = plt.subplots()
    fix.set_size_inches(25,8)

    axs.imshow(grid.cpu().numpy().transpose(1,2,0))
    axs.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
#     fix.savefig(f"{name}.pdf", format="pdf", bbox_inches="tight")
#     fix.show()

def visualize_results(model, device):
    test_input, test_target = next(iter(train_loader))
    
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    
    model.eval() 
    
    predictions = model(test_input)
    predictions = predictions.to(device)
    print(predictions.shape)
    grid_gt = make_grid(test_target[0])
    show(grid_gt, "gt")

    grid_out = make_grid(predictions[0])
    show(grid_out, "output")
#     grid_out = make_grid(predictions[0])
    
if __name__ == "__main__":
    # Get test_loader
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_set = MNIST_Moving(root='.data/mnist', train=True, download=True)
    test_set = MNIST_Moving(root='.data/mnist', train=False, download=True)

    batch_size = 1

    train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=batch_size,
                    shuffle=False)
    test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)
    # Define model
    model = Model()
    
    # Load model
    model, _, _  = utils.loading_model(model, "models/model_7.pth")

    # Visualize
    visualize_results(model, test_loader, device)