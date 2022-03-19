import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import utils 
from model import Model
from dataset import MNIST_Moving
import sys 
import torchvision.transforms as transforms

def show(grid, name):
    fix, axs = plt.subplots()
    fix.set_size_inches(25,8)

    axs.imshow(grid.cpu().numpy().transpose(1,2,0))
    axs.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fix.savefig(f"{name}.png", format="png", bbox_inches="tight")
#     fix.show()

def visualize_results(model, test_loader, device):
    test_input, test_target = next(iter(test_loader))
    
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    
    full_gt_seq = torch.cat((test_input, test_target), dim=1)

    model.eval() 
    with torch.no_grad():
        predictions = model(test_input)
        predictions = predictions.to(device)
    
    grid_gt = make_grid(full_gt_seq[0])
    show(grid_gt, "gt")

    grid_out = make_grid(predictions[0])
    show(grid_out, "output")
#     grid_out = make_grid(predictions[0])



if __name__ == "__main__":
    # Get test_loader
    if len(sys.argv) != 2:
        print("Please add model name")
        sys.exit()

    model_path = sys.argv[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(),
                transforms.Resize((64, 64)),
                ])

    test_set = MNIST_Moving(root='.data/mnist', train=True, download=True, transform=transform, target_transform=transform)

    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)
    # Define model
    model = Model()
    
    # Load model
    model, _, _  = utils.loading_model(model, model_path)
    model = model.to(device)
    # Visualize
    visualize_results(model, test_loader, device)
