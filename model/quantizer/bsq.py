import torch
from model.quantizer.bsq_util import BinarySphericalQuantizer
from typing import Tuple, Mapping, Text
import torch.nn.functional as F

class BSQQuantizer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bsq = BinarySphericalQuantizer(
            embed_dim=config.model.titok.token_size,
            beta=1.0,
            gamma0=config.model.titok.persample_entropy_weight,
            gamma=config.model.titok.cb_entropy_weight, 
            zeta=1.0,
            soft_entropy=config.model.titok.soft_entropy,
            group_size=config.model.titok.embed_group_size,
            persample_entropy_compute=config.model.titok.persample_entropy_compute,
            cb_entropy_compute=config.model.titok.cb_entropy_compute,
            input_format='blc',
            l2_norm=config.model.titok.post_q_l2_norm,
            inv_temperature=1.0,
        )

        self.l2_norm = config.model.titok.l2_norm

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]: # in = [B, L, C]
        if self.l2_norm:
            z = F.normalize(z, dim=-1)

        return self.bsq(z)

    def indices_to_codes(self, indices):
        return self.bsq.indexes_to_codes(indices)