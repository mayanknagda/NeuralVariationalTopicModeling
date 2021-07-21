# Import Libs and Packages
import torch
from datetime import datetime
import os
import logging
from data.loaddata import text_data
from model.vae import VAE
from training import fit
from config import params
from visualization import plot

# Import Params
dir_name = params['dir_name'] # Name of directory for logging outputs in the output/ folder
batch_size = params['batch_size'] # Batch size for traininig
seed = params['seed'] # Seed to replicate the experiments
lr = params['lr'] # Learining Rate used for training,
weight_decay = params['weight_decay'] # Regularization parameter (if regularization is used in the optimizer function
beta_for_vae = params['beta'] # beta for beta - VAE
dropout = params['dropout'] # Dropout percentage (torch)
optimizer = params['optimizer'] # Optimizer function
filename = params['filename'] # Filename of the document in data/ you want to run the code on
hidden_size = params['hidden_size'] # Our VAE encoder has 3 layers, edit the hidden size of each layer you want it to be
num_topics = params['num_topics'] # Number of topics you want to extract from the code
epochs = params['epochs'] # Total epochs you want to run your experiment on
model_type = params['model_type'] # Type of model you want to use (LogNormal, Dirichlet, Weibull)
train_idx = params['train_idx'] # Start training from the document number idx of the CSV
test_idx = params['test_idx'] # Include the last x documents in the CSV for testing and validation (50-50)
val_idx = int(test_idx/2) # Include the last x documents in the CSV for validation

# Make the Data
vocab, docs, train_dl, val_dl, test_dl, bow, texts = text_data(filename=filename, train_idx=train_idx, test_idx=test_idx, val_idx=val_idx, batch_size=batch_size)
vocab_size = docs.shape[1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed)

# Declaring model and optimizer
model = VAE(vocab_size, hidden_size, num_topics, dropout, model_type, beta_for_vae)
model = model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)

#  Making Folder for storing outputs
now = datetime.now()
path = now.strftime("output/%d.%m.%y.%H.%M.%S.") + dir_name
os.mkdir(path)

# Run, trainings
beta, history = fit(epochs, train_dl, val_dl, model, optim, device, path)

# Visualization
plot(beta, vocab, path, num_topics)

# Logging

# Create a log file in the folder
log_filename = path + '/logfile.log'
open(log_filename[0:], 'a').close()

# Gets or creates a logger
logger = logging.getLogger(__name__)

# set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
file_handler = logging.FileHandler(log_filename[0:])
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)
logger.info('---Start Logging---')
logger.info(
    'Dataset: {}. Total Epochs: {}. Total Topics: {}. Model Type: {}. lr: {}. seed: {}.'.format(filename, epochs, num_topics, model_type, lr, seed))
for hist in history:
    logger.info(hist)
logger.info('---Stop Logging---')
logger.removeHandler(file_handler)
del logger, file_handler

## To Be Continued...