import torch
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.gamma import Gamma
from torch.distributions.weibull import Weibull
import torch.nn as nn
from torch.distributions import LogNormal

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
        hidden_layer_two = activation(self.linear_layer_two(hidden_layer_one))
        hidden_layer_three = activation(self.linear_layer_three(hidden_layer_two))
        dropout_layer = self.drop(hidden_layer_three)
        return dropout_layer

    
class DecoderModule(nn.Module):
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.topics_to_doc = nn.Linear(num_topics, vocab_size)
        self.batch_normalization = nn.Linear(vocab_size)
        
    def forward(self, inputs):
        return nn.LogSoftmax(self.batch_normalization(self.topics_to_doc(inputs)), dim=1)


class EncoderToLogNormal(nn.Module):
    def __init__(self, hidden_size, num_topics):
        super().__init__()
        self.linear_mean = nn.Linear(hidden_size[2], num_topics)
        self.linear_var = nn.Linear(hidden_size[2], num_topics)
        self.batch_norm_mean = nn.BatchNorm1d(num_topics, affine=False)
        self.batch_norm_var = nn.BatchNorm1d(num_topics, affine=False)
    
    def forward(self, hidden):
        mean = self.batch_norm_mean(self.linear_mean(hidden))
        var = self.batch_norm_var(self.linear_var(hidden))
        dist = LogNormal(mean, var.exp())
        return dist
        

class EncoderToDirichlet(nn.Module):
    def __init__(self, hidden_size, num_topics):
        super().__init__()
        self.linear_alpha = nn.Linear(hidden_size[2], num_topics)
        self.batch_norm_alpha = nn.BatchNorm1d(num_topics, affine=False)
    
    def forward(self, hidden):
        alpha = self.batch_norm_alpha(self.linear_alpha(hidden))
        dist = Dirichlet(alpha)
        return dist

    
class EncoderToWeibull(nn.Module):
    def __init__(self, hidden_size, num_topics):
        super().__init__()
        self.linear_lambda = nn.Linear(hidden_size[2], num_topics)
        self.linear_k = nn.Linear(hidden_size[2], num_topics)
        self.batch_norm_lambda = nn.BatchNorm1d(num_topics, affine=False)
        self.batch_norm_k = nn.BatchNorm1d(num_topics, affine=False)
    
    def forward(self, hidden):
        lambda_ = self.batch_norm_lambda(self.linear_lambda(hidden))
        k = self.batch_norm_k(self.linear_k(hidden))
        dist = Weibull(lambda_, k)
        return dist

    
class EncoderToGamma(nn.Module):
    def __init__(self, hidden_size, num_topics):
        super().__init__()
        self.linear_k = nn.Linear(hidden_size[2], num_topics)
        self.linear_theta = nn.Linear(hidden_size[2], num_topics)
        self.batch_norm_k = nn.BatchNorm1d(num_topics, affine=False)
        self.batch_norm_theta = nn.BatchNorm1d(num_topics, affine=False)
    
    def forward(self, hidden):
        k = self.batch_norm_k(self.linear_k(hidden))
        theta = self.batch_norm_theta(self.linear_theta(hidden))
        dist = Gamma(k, theta)
        return dist


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        pass