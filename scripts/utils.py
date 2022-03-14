import os
import torch 
import numpy as np
from tqdm import tqdm

def train_epoch(model, train_loader, optimizer, criterion, epoch, device):
    """ Training a model for one epoch """
    
    loss_list = []
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, full_seq in progress_bar:

        full_seq = full_seq.type(torch.FloatTensor).to(device)
        
        
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        
        # Forward pass only to get logits/output
        outputs = model(full_seq)
        outputs = outputs.to(device)
         
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, full_seq)
        loss_list.append(loss.item())
        
        # Getting gradients w.r.t. parameters
        loss.backward()
         
        # Updating parameters
        optimizer.step()
        
        progress_bar.set_description(f"Epoch {epoch+1} Iter {i+1}: loss {loss.item():.5f}. ")
        
    mean_loss = np.mean(loss_list)
    return mean_loss, loss_list


@torch.no_grad()
def eval_model(model, eval_loader, criterion, device):
    """ Evaluating the model for either validation or test """
    loss_list = []
    pbar = tqdm(enumerate(eval_loader), total=len(eval_loader))
    for idx, full_seq in pbar:

        full_seq = full_seq.type(torch.FloatTensor).to(device)
        target_seq = full_seq[:, 10:, :, :, :]
        
        
        # Forward pass only to get logits/output
        outputs = model(full_seq)
        outputs = outputs.to(device)
        
        preds = outputs[:, 10:, :, :, :]
        
        loss = criterion(preds, target_seq)

        loss_list.append(loss.item())
            
        pbar.set_description(f"Test loss: loss {loss.item():.2f}")
    mean_loss = np.mean(loss_list)
    
    return mean_loss


def train_model(model, optimizer, scheduler, criterion, train_loader,\
                valid_loader, num_epochs, device):
    """ Training a model for a given number of epochs"""
    
    train_loss = []
    val_loss =  []
    loss_iters = []
    epochs = []
    
    for epoch in range(num_epochs):
           
        # validation epoch
        model.eval()  # important for dropout and batch norms
        loss = eval_model(
                    model=model, eval_loader=valid_loader,
                    criterion=criterion, device=device
            )
        val_loss.append(loss)
        epochs.append(epoch+1)
        
        # training epoch
        model.train()  # important for dropout and batch norms
        mean_loss, cur_loss_iters = train_epoch(
                model=model, train_loader=train_loader, optimizer=optimizer,
                criterion=criterion, epoch=epoch, device=device
            )
        scheduler.step()
        train_loss.append(mean_loss)
        loss_iters = loss_iters + cur_loss_iters
        

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"    Train loss: {round(mean_loss, 5)}")
        print(f"    Valid loss: {round(loss, 5)}")
        print("\n")
        saving_model(model, optimizer, epoch)

    print(f"Training completed")
    return train_loss, val_loss, loss_iters, epochs


def saving_model(model, optimizer, epoch):
    if not os.path.exists("models"):
        os.makedirs("models")
    save_path = f"models/model_{epoch+1}.pth"
    torch.save({
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'stats': stats
    }, save_path)
    

def loading_model(model, path):
    optimizer = torch.optim.Adam(params=model.parameters(), lr= 3e-4)
    checkpoint =  torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    # stats = checkpoint['stats']
    return model, optimizer, epoch