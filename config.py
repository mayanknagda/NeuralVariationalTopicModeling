import torch
params = {
    'dir_name': 'Wiki_clean_more_topics_dirichlet_50epoch', # Name of directory for logging outputs in the output/ folder
    'batch_size': 32, # Batch size for traininig
    'seed': 3, # Seed to replicate the experiments
    'lr': 1e-3, #Learining Rate used for training,
    'weight_decay': 0, # Regularization parameter (if regularization is used in the optimizer function)
    'dropout': 0.2, # Dropout percentage (torch)
    'optimizer': torch.optim.Adam, # Optimizer function
    'filename': '20ng.clean.txt', # Filename of the document in data/ you want to run the code on
    'hidden_size': [512, 256, 128], # Our VAE encoder has 3 layers, edit the hidden size of each layer you want it to be
    'num_topics': 30, # Number of topics you want to extract from the code
    'epochs': 100, # Total epochs you want to run your experiment on
    'model_type': 2, # Type of model you want to use (1 - LogNormal, 2 - Dirichlet, 3 - Weibull)
    'train_idx': 0, # Start training from the document number idx of the CSV
    'test_idx': 3000, # Include the last x documents in the CSV for testing
}