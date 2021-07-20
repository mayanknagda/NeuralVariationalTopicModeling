import torch

def train_one_epoch(train_dl, model, optim, device):
    model.train()
    epoch_total_loss, epoch_nll_loss, epoch_kld_loss = [], [], []
    for batch in train_dl:
        batch_total_loss, batch_nll_loss, batch_kld_loss = train_one_batch(batch, model, optim, device)
        epoch_total_loss.append(batch_total_loss), epoch_nll_loss.append(batch_nll_loss), epoch_kld_loss.append(batch_kld_loss)
    loss ={'total_loss': torch.mean(epoch_total_loss), 'nll_loss': torch.mean(epoch_nll_loss), 'kld_loss': torch.mean(epoch_kld_loss)}
    return loss

def train_one_batch(batch, model, optim, device):
    batch.to(torch.device(device))
    optim.zero_grad()
    out, posterior = model(batch[0])
    nll, kld = model.loss(out, batch[0], posterior)
    loss = nll + kld
    loss.backward()
    optim.step()
    return loss.item(), nll.item(), kld.item()

def validate_one_epoch(val_dl, model, device):
    model.eval()
    epoch_total_loss, epoch_nll_loss, epoch_kld_loss = [], [], []
    for batch in val_dl:
        batch_total_loss, batch_nll_loss, batch_kld_loss = validate_one_batch(batch, model, device)
        epoch_total_loss.append(batch_total_loss), epoch_nll_loss.append(batch_nll_loss), epoch_kld_loss.append(batch_kld_loss)
    loss ={'total_loss': torch.mean(epoch_total_loss), 'nll_loss': torch.mean(epoch_nll_loss), 'kld_loss': torch.mean(epoch_kld_loss)}
    return loss

def validate_one_batch(batch, model, device):
    batch.to(torch.device(device))
    out, posterior = model(batch[0])
    nll, kld = model.loss(out, batch[0], posterior)
    loss = nll + kld
    return loss.item(), nll.item(), kld.item()

def fit(epochs, train_dl, val_dl, model, optim, device, path):
    history = []
    for i in range(epochs):
        epoch_train_loss = train_one_epoch(train_dl, model, optim, device)
        epoch_validation_loss = validate_one_epoch(val_dl, model, device)
        print('Epoch: {}, Train Loss: {} (NLL: {}, KLD: {}), Validation Loss: {} (NLL: {}, KLD: {})'.format())
        history.append({'epoch': '', 'train_loss': '', 'train_loss_nll': '', 'train_loss_kld': '', 'val_loss': '', 'val_loss_nll': '', 'val_loss_kld': ''})