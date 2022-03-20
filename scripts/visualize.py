import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

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