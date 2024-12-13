"""This files contains training loss implementation.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

Ref:
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
"""
from typing import Mapping, Text, Tuple

import torch
import torch.nn as nn
    
class ReconstructionLoss(torch.nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.config = config

        self.codebook_rampup_steps = config.model.titok.get("codebook_rampup_steps", 0)
        self.codebook_rampup_multiplier = config.model.titok.get("codebook_rampup_multiplier", 3.0)

    def forward(self,
                target: torch.Tensor,
                recon: torch.Tensor,
                extra_results_dict: Mapping[Text, torch.Tensor],
                global_step: int,
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        return self._forward_generator(target, recon, extra_results_dict, global_step)

    def _forward_generator(self,
                           target: torch.Tensor,
                           recon: torch.Tensor,
                           extra_results_dict: Mapping[Text, torch.Tensor],
                           global_step: int,
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        loss_dict = {}

        recon = recon.contiguous()
        loss_fct = nn.MSELoss(reduction="mean")

        assert recon.shape == target.shape, f"Prediction and target shapes must match: R={recon.shape} | T={target.shape} "

        reconstruction_loss = loss_fct(recon, target)
        
        if "kl_loss" in extra_results_dict.keys():
            kl_loss = extra_results_dict["kl_loss"]
            total_loss = reconstruction_loss + self.config.model.titok.kl_weight * kl_loss
            loss_dict["kl_loss"] = kl_loss.detach()
            loss_dict["total_loss"] = total_loss.detach()

        elif "ent_loss" in extra_results_dict.keys():
            ent_loss = extra_results_dict["ent_loss"]
            commit_loss = extra_results_dict["commit_loss"]

            if self.codebook_rampup_steps > 0:
                rampup_rate =  min(self.codebook_rampup_steps, global_step) / self.codebook_rampup_steps
                scaled_codebook_weight = self.config.model.titok.ent_loss_weight * (1.0 * rampup_rate + self.codebook_rampup_multiplier * (1 - rampup_rate))
            else:
                scaled_codebook_weight = self.config.model.titok.ent_loss_weight

            total_loss = reconstruction_loss + scaled_codebook_weight * (ent_loss + commit_loss * self.config.model.titok.commit_loss_weight)
            
            loss_dict["ent_loss"] = ent_loss.detach()
            loss_dict["commit_loss"] = commit_loss.detach()
            loss_dict["total_loss"] = total_loss.detach()
            
        else:
           total_loss = reconstruction_loss
           loss_dict["total_loss"] = reconstruction_loss.detach() # for FSQ, no aux loss


        loss_dict["reconstruction_loss"] = reconstruction_loss.detach()

        return total_loss, loss_dict

    
