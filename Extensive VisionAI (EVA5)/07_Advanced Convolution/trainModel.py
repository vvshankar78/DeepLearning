def train(model, device, train_loader, optimizer, epoch, loss_type, plot_flag, plot_after_epochs):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    count = 0
    loss = 0
    
    train_losses = []
    train_acc = []
    
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)
    
        # Init
        optimizer.zero_grad()
    
        # Predict
        y_pred = model(data)
        # print(y_pred)
    
        # Calculate loss
        loss = F.nll_loss(y_pred, target)
        # print(loss)
            
        # Backpropagation
        loss.backward()
        optimizer.step()
    
        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    
        x = torch.tensor((~pred.eq(target.view_as(pred))), dtype=torch.float, device = device).clone().detach().requires_grad_(True)      
    
    # loss /= len(train_loader.dataset)
    train_losses.append(loss)
    train_acc.append(100*correct/processed)
        
    return (train_acc,train_losses)