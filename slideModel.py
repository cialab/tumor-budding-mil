import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Attention(nn.Module):
    def __init__(self,ps):
        super(Attention, self).__init__()
        self.L = 1000
        self.D = 100
        self.K = 1
        self.P = 1000

        self.fe1 = models.squeezenet1_0(pretrained=True)

        self.fe2 = nn.Sequential(
            nn.Linear(self.P, self.L),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.L,self.L),
            nn.ReLU(),
            nn.Dropout()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)
        
        H = self.fe1(x)
        H = self.fe2(H)
        
        A = self.attention(H)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)

        M = torch.mm(A, H)

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        
        return Y_prob, Y_hat, A

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, Y_hat, A = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        return neg_log_likelihood, error, Y_hat, Y_prob, A