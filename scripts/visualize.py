import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def show(grid, name):
    fix, axs = plt.subplots()
    fix.set_size_inches(25,8)

    axs.imshow(grid.cpu().numpy().transpose(1,2,0))
    axs.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fix.savefig(f"{name}.pdf", format="pdf", bbox_inches="tight")
    fix.show()
        
def visualize_results(model, test_loader, device):
    test_input = next(iter(test_loader)).type(torch.FloatTensor).to(device)
    model.eval() 
    out_ = model(test_input)
    out_ = out_.to(device)

    grid_gt = make_grid(test_input[0])
    grid_out = make_grid(out_[0])
    show(grid_gt, "gt")
    show(grid_out, "output")