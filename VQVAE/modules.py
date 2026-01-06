import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import numpy as np

from functions import vq, vq_st

def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VAE(nn.Module):
    def __init__(self, input_dim, dim, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, z_dim * 2, 3, 1, 0),
            nn.BatchNorm2d(z_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, dim, 3, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)

        q_z_x = Normal(mu, logvar.mul(.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()

        x_tilde = self.decoder(q_z_x.rsample())
        return x_tilde, kl_div


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar, indices


class VectorQuantizedVAE(nn.Module):
    def __init__(self, state_dim, seq_len=4, K=10, dim=32, output_dim=2):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim+seq_len*output_dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True), 
            nn.Linear(dim, dim), # size = (B, dim)
        )

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            nn.Linear(state_dim+dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, seq_len*output_dim),
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, state, x):
        # x = x.view(x.size(0), -1)
        z_e_x = self.encoder(torch.concat((state, x), dim=1))
        z_e_x = z_e_x.unsqueeze(2).unsqueeze(3)  # z_e_x needs to be (B, D, H, W) -> (B, D, 1, 1) for sequence
        z_q_x_st, z_q_x, indices = self.codebook.straight_through(z_e_x)
        z_q_x_st = z_q_x_st.squeeze(2).squeeze(2)

        x_tilde = self.decoder(torch.concat((state, z_q_x_st), dim=1))
        
        return x_tilde, z_e_x, z_q_x, indices
    
    @torch.no_grad()
    def reinit_unused_codes(self, codebook_usage):
        """
        Re-initialize unused vectors according to the likelihood of used ones.
        :param codebook_usage: (n, ) where n is the codebook size, distribution probability of codebook usage.
        """

        device = codebook_usage.device
        n = codebook_usage.shape[0]

        # compute unused codes
        unused_codes = torch.nonzero(torch.eq(codebook_usage, torch.zeros(n, device=device, dtype=codebook_usage.dtype))).squeeze(1)
        n_unused = unused_codes.shape[0]
        if n_unused == 0:
            return

        print("Reinitializing unused codes:")
        print(unused_codes)

        # sample according to most used codes.
        replacements = torch.multinomial(codebook_usage, n_unused, replacement=True)

        # update unused codes
        new_codes = self.codebook.embedding.weight[replacements]
        self.codebook.embedding.weight[unused_codes] = new_codes
    
    @torch.no_grad()
    def forward_decoder(self, state, k_idx):
        """
        k_idx: (B, )
        """
        z_q_x = torch.index_select(self.codebook.embedding.weight, dim=0, index=k_idx)  # (B, dim)
        x_tilde = self.decoder(torch.concat((state, z_q_x), dim=1))
        # restore x_tilde to the original shape: (B, seq_len*output_dim) -> (B, seq_len, output_dim)
        x_tilde = x_tilde.view(-1, self.seq_len, self.output_dim)
        return x_tilde.cpu().detach().numpy()

