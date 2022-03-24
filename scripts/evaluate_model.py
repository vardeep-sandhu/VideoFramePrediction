import torch
import numpy as np
from model import Model
from utils import loading_model
import argparse
from dataset.test_moving_mnist import MovingMNIST_test
from dataset.kth import KTH
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import piqa
from skimage.metrics import structural_similarity 

def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset', '-d', type=str)
    args = parser.parse_args()
    # cfg = AttrDict(utils.load_cfg(args.cfg_name))
    return args

def mse_metric(x1, x2):
    mse = torch.zeros((len(x1)))
    for idx in range(len(x1)):
        mse[idx] =  torch.nn.functional.mse_loss(x1[idx], x2[idx])
    return mse

def mae_metric(x1, x2):
    mae = torch.zeros((len(x1)))
    for idx in range(len(x1)):
        mae[idx] =  torch.nn.functional.l1_loss(x1[idx], x2[idx])
    return mae

def lpips_metric(lpips_criterion, x1, x2):
    # Since LPIPS can be calculated with 3 channels so we repeat the channels 
    x1 = x1.repeat(1, 3, 1, 1)
    x2 = x2.repeat(1, 3, 1, 1)
    lpips = torch.zeros((len(x1)))
    for idx in range(len(x1)):
        lpips[idx] =  lpips_criterion(x1[idx].unsqueeze(0), x2[idx].unsqueeze(0))
    return lpips

def ssim_metric(x1, x2):
    ssim = torch.zeros((len(x1)))
    for idx in range(len(x1)):
        ssim[idx] =  structural_similarity(np.squeeze(x1[idx]), np.squeeze(x2[idx]))
    return ssim

def evaluate_metrices(prediction, target):
    batch_size = target.shape[0]
    mae = []
    mse = []
    psnr = []
    lpips = []
    ssim = []

    for b_inx in range(batch_size):

        psnr_seq = piqa.psnr.psnr(prediction[b_inx], target[b_inx])
        mse_seq = mse_metric(prediction[b_inx], target[b_inx])
        mae_seq = mae_metric(prediction[b_inx], target[b_inx])
        lpips_criterion = piqa.lpips.LPIPS().cuda()
        lpips_seq = lpips_metric(lpips_criterion, prediction[b_inx], target[b_inx])
        ssim_seq = ssim_metric(prediction[b_inx].cpu().numpy(), target[b_inx].cpu().numpy())

        mae.append(mae_seq)
        mse.append(mse_seq)
        psnr.append(psnr_seq)
        lpips.append(lpips_seq)
        ssim.append(ssim_seq)

    return torch.stack(mae), torch.stack(mse), torch.stack(psnr), torch.stack(lpips), torch.stack(ssim) 
    


def main():
    args = parse_commandline()
    device = "cuda"
    model_path = "models/model_3.pth"
    model = Model()
    model, _, _ = loading_model(model, model_path)
    model = model.to(device)

    transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Resize((64, 64)),
                    ])

    if args.test_dataset == "moving_mnist":
        test_set = MovingMNIST_test(root='data/mnist', train=False, download=True, transform=transform, target_transform=transform)
    if args.test_dataset == "kth":
        test_set = KTH("data/kth", transform=transform, download=False, train=False)


    test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=False)

    with torch.no_grad():
        mae_all = []
        mse_all = []
        psnr_all = []
        lpips_all = []
        ssim_all = []

        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for idx, (seq, target) in pbar:
            seq = seq.type(torch.FloatTensor).to(device)
            target = target.type(torch.FloatTensor).to(device)
            
            predictions = model(seq)
            predictions = predictions.to(device)
                    
            mae, mse, psnr, lpips, ssim  = evaluate_metrices(predictions[:, 10:, :, :, :], target)
            mae_all.append(mae)
            mse_all.append(mse)
            psnr_all.append(psnr)
            lpips_all.append(lpips)
            ssim_all.append(ssim)

            # pbar.set_description(f"Test loss: loss {loss.item():.2f}")
        mae_all, mse_all, psnr_all, lpips_all, ssim_all = torch.stack(mae_all), torch.stack(mse_all), torch.stack(psnr_all), torch.stack(lpips_all), torch.stack(ssim_all) 
        print(mae_all.shape)
        return mae_all, mse_all, psnr_all, lpips_all, ssim_all 


if __name__ == "__main__":
    main()