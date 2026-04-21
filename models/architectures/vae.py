import torch
import torch.nn as nn

from globals.config import INPUT_VARS, LATENT_DIM, ENCODER_HIDDEN, DECODER_HIDDEN


## Encoder ##
## Per-depth MLP -> masked mean-pool -> (mu, log_var)
## Input:  profile (batch, depth, n_vars), mask (batch, depth, n_vars)
## Output: mu (batch, latent_dim), log_var (batch, latent_dim)

class Encoder(nn.Module):

    def __init__(self, n_vars=None, latent_dim=LATENT_DIM, hidden=ENCODER_HIDDEN):
        super().__init__()
        n_vars = n_vars or len(INPUT_VARS)

        layers = []
        in_dim = n_vars
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h

        self.mlp    = nn.Sequential(*layers)
        self.fc_mu  = nn.Linear(in_dim, latent_dim)
        self.fc_var = nn.Linear(in_dim, latent_dim)

    def forward(self, profile, mask):
        """
        profile : (batch, depth, n_vars)
        mask    : (batch, depth, n_vars) bool — True where data is real
        Returns : mu (batch, latent_dim), log_var (batch, latent_dim)
        """
        h = self.mlp(profile)                        # (batch, depth, hidden[-1])

        depth_mask = mask.any(dim=-1, keepdim=True)  # (batch, depth, 1)
        depth_mask = depth_mask.float()

        # masked mean-pool over depth
        h = (h * depth_mask).sum(dim=1) / depth_mask.sum(dim=1).clamp(min=1)
                                                     # (batch, hidden[-1])

        mu      = self.fc_mu(h)                      # (batch, latent_dim)
        log_var = self.fc_var(h)                     # (batch, latent_dim)

        return mu, log_var


## Decoder ##
## Unchanged from autoencoder — takes a latent vector, returns reconstruction.
## Input:  p (batch, latent_dim), depth_levels (depth,)
## Output: reconstruction (batch, depth, n_vars)

class Decoder(nn.Module):

    def __init__(self, n_vars=None, latent_dim=LATENT_DIM, hidden=DECODER_HIDDEN):
        super().__init__()
        n_vars = n_vars or len(INPUT_VARS)

        layers = []
        in_dim = latent_dim + 1    # +1 for depth in meters
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, n_vars)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, p, depth_levels):
        """
        p            : (batch, latent_dim)
        depth_levels : (depth,) tensor of depth values in meters
        Returns      : (batch, depth, n_vars)
        """
        batch = p.shape[0]
        depth = depth_levels.shape[0]

        p_expanded = p.unsqueeze(1).expand(-1, depth, -1)
        d          = depth_levels.view(1, -1, 1).expand(batch, -1, -1)
        inp        = torch.cat([p_expanded, d], dim=-1)

        return self.mlp(inp)                         # (batch, depth, n_vars)


## VAE ##
## Wraps encoder + decoder.
## forward() returns (recon, mu, log_var) so the training loop can compute
## the combined loss: reconstruction + KL divergence.

class VAE(nn.Module):

    def __init__(self, n_vars=None, latent_dim=LATENT_DIM,
                 encoder_hidden=ENCODER_HIDDEN, decoder_hidden=DECODER_HIDDEN):
        super().__init__()
        n_vars = n_vars or len(INPUT_VARS)
        self.encoder = Encoder(n_vars, latent_dim, encoder_hidden)
        self.decoder = Decoder(n_vars, latent_dim, decoder_hidden)

    def reparameterize(self, mu, log_var):
        """
        Sample z = mu + eps * std  where  eps ~ N(0, 1).
        Using log_var for numerical stability — std = exp(0.5 * log_var).
        Only sample during training; use mu directly at eval time.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, profile, mask, depth_levels):
        """
        profile      : (batch, depth, n_vars)
        mask         : (batch, depth, n_vars)
        depth_levels : (depth,) tensor of depth values in meters
        Returns      : recon (batch, depth, n_vars), mu (batch, latent_dim),
                       log_var (batch, latent_dim)
        """
        mu, log_var = self.encoder(profile, mask)
        z           = self.reparameterize(mu, log_var)
        recon       = self.decoder(z, depth_levels)
        return recon, mu, log_var
    def encode_mu(self, profile, mask):
        mu, _ = self.encoder(profile, mask)
        return mu
    def save(self, path, stats=None):
        torch.save({"model_state": self.state_dict(), "stats": stats}, path)
        print(f"Saved VAE to {path}")

    @classmethod
    def load(cls, path, device="cpu", **kwargs):
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = cls(**kwargs)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        return model, ckpt.get("stats")


## Loss ##
## Call this in your training loop instead of plain MSE.
## beta controls the weight of the KL term — start with 1.0.
## beta < 1 is "beta-VAE" which encourages more disentangled latent dims.

def vae_loss(recon, target, mask, mu, log_var, beta=1.0):
    """
    recon   : (batch, depth, n_vars)
    target  : (batch, depth, n_vars)
    mask    : (batch, depth, n_vars) bool — only penalize observed values
    mu      : (batch, latent_dim)
    log_var : (batch, latent_dim)
    """
    # reconstruction loss — masked MSE over observed entries only
    diff    = (recon - target) ** 2
    recon_loss = (diff * mask.float()).sum() / mask.float().sum().clamp(min=1)

    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + beta * kl_loss, recon_loss, kl_loss

