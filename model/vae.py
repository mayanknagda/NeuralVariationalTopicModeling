import torch
import torch.nn as nn
from torch.distributions import LogNormal, Dirichlet, Gamma, Laplace
from torch.distributions import kl_divergence

class EncoderModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_layer_one = nn.Linear(vocab_size, hidden_size[0])
        self.linear_layer_two = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear_layer_three = nn.Linear(hidden_size[1], hidden_size[2])
        
    def forward(self, inputs):
        activation = nn.LeakyReLU()
        hidden_layer_one = activation(self.linear_layer_one(inputs))
        hidden_layer_two = self.dropout(activation(self.linear_layer_two(hidden_layer_one)))
        hidden_layer_three = self.dropout(activation(self.linear_layer_three(hidden_layer_two)))
        return hidden_layer_three


class DecoderModule(nn.Module):
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.topics_to_doc = nn.Linear(num_topics, vocab_size)
        self.batch_normalization = nn.BatchNorm1d(vocab_size, affine=False)
        
    def forward(self, inputs):
        log_softmax = nn.LogSoftmax(dim = 1)
        return log_softmax(self.batch_normalization(self.topics_to_doc(inputs)))


class EncoderToLogNormal(nn.Module):
    def __init__(self, hidden_size, num_topics):
        super().__init__()
        self.linear_mean = nn.Linear(hidden_size[2], num_topics)
        self.linear_var = nn.Linear(hidden_size[2], num_topics)
        self.batch_norm_mean = nn.BatchNorm1d(num_topics, affine=False)
        self.batch_norm_var = nn.BatchNorm1d(num_topics, affine=False)
    
    def forward(self, hidden):
        mean = self.batch_norm_mean(self.linear_mean(hidden))
        var = 0.5 * self.batch_norm_var(self.linear_var(hidden))
        dist = LogNormal(mean, var.exp())
        return dist
        

class EncoderToDirichlet(nn.Module):
    def __init__(self, hidden_size, num_topics):
        super().__init__()
        self.linear_alpha = nn.Linear(hidden_size[2], num_topics)
        self.batch_norm_alpha = nn.BatchNorm1d(num_topics, affine=False)
    
    def forward(self, hidden):
        alpha = self.batch_norm_alpha(self.linear_alpha(hidden)).exp().cpu()
        dist = Dirichlet(alpha)
        return dist


class EncoderToLaplace(nn.Module):
    def __init__(self, hidden_size, num_topics):
        super().__init__()
        self.linear_lambda = nn.Linear(hidden_size[2], num_topics)
        self.linear_k = nn.Linear(hidden_size[2], num_topics)
        self.batch_norm_lambda = nn.BatchNorm1d(num_topics, affine=False)
        self.batch_norm_k = nn.BatchNorm1d(num_topics, affine=False)
    
    def forward(self, hidden):
        loc = self.batch_norm_lambda(self.linear_lambda(hidden))
        scale = 0.5 * self.batch_norm_k(self.linear_k(hidden))
        dist = Laplace(loc, scale.exp())
        return dist


class EncoderToGamma(nn.Module):
    def __init__(self, hidden_size, num_topics):
        super().__init__()
        self.linear_k = nn.Linear(hidden_size[2], num_topics)
        self.linear_theta = nn.Linear(hidden_size[2], num_topics)
        self.batch_norm_k = nn.BatchNorm1d(num_topics, affine=False)
        self.batch_norm_theta = nn.BatchNorm1d(num_topics, affine=False)
    
    def forward(self, hidden):
        k = 0.5 * self.batch_norm_k(self.linear_k(hidden))
        theta = 0.5 * self.batch_norm_theta(self.linear_theta(hidden))
        dist = Gamma(k.exp(), theta.exp())
        return dist


class VAE(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_topics, dropout, model_type, beta):
        super().__init__()
        self.encoder = EncoderModule(vocab_size, hidden_size, dropout)
        if model_type == 1:
            self.encoder_to_dist = EncoderToLogNormal(hidden_size, num_topics)
        elif model_type == 2:
            self.encoder_to_dist = EncoderToDirichlet(hidden_size, num_topics)
        elif model_type == 3:
            self.encoder_to_dist = EncoderToLaplace(hidden_size, num_topics)
        elif model_type == 4:
            self.encoder_to_dist = EncoderToGamma(hidden_size, num_topics)
        self.decoder = DecoderModule(vocab_size, num_topics, dropout)
        self.beta = beta
        
    def forward(self, inputs):
        encoder_output = self.encoder(inputs)
        dist = self.encoder_to_dist(encoder_output)
        if self.training:
            dist_to_decoder = dist.rsample().to(inputs.device)
        else:
            dist_to_decoder = dist.mean.to(inputs.device)
        softmax = nn.Softmax(dim = 1)
        dist_to_decoder = softmax(dist_to_decoder)
        reconstructed_documents = self.decoder(dist_to_decoder)
        return reconstructed_documents, dist
    
    def loss(self, reconstructed, original, posterior): # We need to have NLL Loss as well KLD Loss
        if isinstance(posterior, LogNormal):
            loc = torch.zeros_like(posterior.loc)
            scale = torch.ones_like(posterior.scale)        
            prior = LogNormal(loc, scale)
        elif isinstance(posterior, Dirichlet):
            alphas = torch.ones_like(posterior.concentration) * 0.01
            prior = Dirichlet(alphas)
        elif isinstance(posterior, Laplace):
            loc = torch.zeros_like(posterior.loc)
            scale = torch.ones_like(posterior.scale)
            prior = Laplace(loc, scale)
        elif isinstance(posterior, Gamma):
            concentration = torch.ones_like(posterior.concentration) * 9
            rate = torch.ones_like(posterior.rate) * 0.5
            prior = Gamma(concentration, rate)
            
        NLL = - torch.sum(reconstructed*original)
        KLD = torch.sum(kl_divergence(posterior, prior).to(reconstructed.device))
        loss_for_training = NLL + self.beta * KLD
        return NLL, KLD, loss_for_training
        