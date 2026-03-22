import torch
import torch.nn as nn

from globals.config import LATENT_DIM, ODE_HIDDEN


## ODEFunc ##
## Defines dp/dt = MLP(p, lat, lon)
## lat/lon are carried in the augmented state but frozen (derivative = 0)
## z_aug = cat(p, lat, lon) of size latent_dim + 2

class ODEFunc(nn.Module):

    def __init__(self, latent_dim=LATENT_DIM, hidden=ODE_HIDDEN):
        super().__init__()
        in_dim = latent_dim + 2   # p + lat + lon

        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers += [nn.Linear(in_dim, latent_dim)]

        self.mlp = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, t, z_aug):
        """
        t     : scalar (current integration time)
        z_aug : (batch, latent_dim + 2)
        Returns dz_aug/dt : (batch, latent_dim + 2)
        """
        dp    = self.mlp(z_aug)
        zeros = torch.zeros(z_aug.shape[0], 2, device=z_aug.device)
        return torch.cat([dp, zeros], dim=-1)

    def save(self, path):
        torch.save({"model_state": self.state_dict()}, path)
        print(f"Saved ODEFunc to {path}")

    @classmethod
    def load(cls, path, device="cpu", **kwargs):
        ckpt  = torch.load(path, map_location=device)
        model = cls(**kwargs)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        return model