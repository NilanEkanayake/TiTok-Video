import torch
from torch import nn
import math


class LRSched(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        conf_g = self.config.tokenizer.optimizer
        conf_d = self.config.discriminator.optimizer

        self.max_lr = conf_g.learning_rate
        self.end_lr = conf_g.end_lr

        self.warmup_steps = max(conf_g.warmup_steps, 1)
        self.max_steps = config.training.main.max_steps
        self.schedule = conf_g.lr_schedule

        ### disc
        self.lr_divisor = conf_d.lr_divisor
        self.disc_warm = max(conf_d.warmup_steps, 1)
        self.disc_start = config.discriminator.losses.disc_start
        self.disc_warm_trg = self.calc_lr(self.disc_start+self.disc_warm) / self.lr_divisor
        ###

        self.last_lr = self.calc_lr(0)


    # https://github.com/philippe-eecs/vitok/blob/main/pytorch_vitok_old/utils.py
    def calc_lr(self, step):
        if step < self.warmup_steps:
            # lr = max(self.max_lr * step / self.warmup_steps, self.end_lr) # doesn't warm from 0?
            lr = self.max_lr * step / self.warmup_steps
        elif self.schedule == "constant":
            lr = max(self.max_lr, self.end_lr)
        elif self.schedule == "cosine":
            lr = self.end_lr + (self.max_lr - self.end_lr) * 0.5 * (
                1.0
                + math.cos(math.pi * (step - self.warmup_steps) / (self.max_steps - self.warmup_steps))
            )
        return lr
    
    def forward(self, optimizer, step, disc_step=False):
        if disc_step:
            assert step >= self.disc_start

            lr = self.last_lr / self.lr_divisor
            if step < (self.disc_start+self.disc_warm):
                # lr = max(self.disc_warm_trg * ((step-self.disc_start) / self.disc_warm), self.end_lr) # let start from LR 0 instead?
                lr = self.disc_warm_trg * min((step-self.disc_start) / self.disc_warm, 1.0)
        else:
            lr = self.calc_lr(step)
            self.last_lr = lr

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return lr # for logging