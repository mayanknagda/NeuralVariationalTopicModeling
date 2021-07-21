import torch

def train_one_epoch(train_dl, model, optim, device):
    model.train()
    epoch_total_loss, epoch_nll_loss, epoch_kld_loss = [], [], []
    for batch in train_dl:
        batch_total_loss, batch_nll_loss, batch_kld_loss = train_one_batch(batch, model, optim, device)
        epoch_total_loss.append(batch_total_loss), epoch_nll_loss.append(batch_nll_loss), epoch_kld_loss.append(batch_kld_loss)
    loss ={'total_loss': torch.mean(torch.Tensor(epoch_total_loss)), 'nll_loss': torch.mean(torch.Tensor(epoch_nll_loss)), 'kld_loss': torch.mean(torch.Tensor(epoch_kld_loss))}
    return loss

def train_one_batch(batch, model, optim, device):
    docs = batch[0].to(torch.device(device))
    optim.zero_grad()
    out, posterior = model(docs)
    nll, kld, loss_for_training = model.loss(out, docs, posterior)
    loss = nll + kld
    loss_for_training.backward()
    optim.step()
    return loss.item(), nll.item(), kld.item()

def validate_one_epoch(val_dl, model, device):
    model.eval()
    epoch_total_loss, epoch_nll_loss, epoch_kld_loss = [], [], []
    for batch in val_dl:
        batch_total_loss, batch_nll_loss, batch_kld_loss = validate_one_batch(batch, model, device)
        epoch_total_loss.append(batch_total_loss), epoch_nll_loss.append(batch_nll_loss), epoch_kld_loss.append(batch_kld_loss)
    loss ={'total_loss': torch.mean(torch.Tensor(epoch_total_loss)), 'nll_loss': torch.mean(torch.Tensor(epoch_nll_loss)), 'kld_loss': torch.mean(torch.Tensor(epoch_kld_loss))}
    return loss

def validate_one_batch(batch, model, device):
    docs = batch[0].to(torch.device(device))
    out, posterior = model(docs)
    nll, kld, _ = model.loss(out, docs, posterior)
    loss = nll + kld
    return loss.item(), nll.item(), kld.item()

def fit(epochs, train_dl, val_dl, model, optim, device, path):
    history = []
    for epoch in range(epochs):
        epoch_train_loss = train_one_epoch(train_dl, model, optim, device)
        epoch_validation_loss = validate_one_epoch(val_dl, model, device)
        log = {
                'epoch': epoch + 1,
                'train_loss': epoch_train_loss['total_loss'],
                'train_loss_nll': epoch_train_loss['nll_loss'],
                'train_loss_kld': epoch_train_loss['kld_loss'],
                'val_loss': epoch_validation_loss['total_loss'],
                'val_loss_nll': epoch_validation_loss['nll_loss'],
                'val_loss_kld': epoch_validation_loss['kld_loss']
            }
        history.append(log)
        print(log)
    beta = model.decoder.topics_to_doc.weight.cpu().detach().T
    return beta, history