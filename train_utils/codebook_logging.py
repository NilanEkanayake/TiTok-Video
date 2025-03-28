import torch
from torch import nn
from scipy.stats import entropy

class CodebookLogger(nn.Module):
    def __init__(self, codebook_size):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_frequencies = torch.zeros(codebook_size)
        self.codebook_counter = 0
        self.codebook_indices = []

    def forward(self, codes):
        for sample in codes:
            if len(self.codebook_indices) == self.codebook_size:
                self.codebook_indices.pop(0)
            self.codebook_indices.append(sample)

    def get_scores(self):
        if self.is_score_ready():
            code_frequencies = torch.zeros(self.codebook_size)
            for sample in self.codebook_indices:
                code_frequencies += torch.bincount(sample, minlength=self.codebook_size).cpu()

            freq_np = code_frequencies.float().numpy()
            codebook_dict = {
                'codebook/usage_percent': (code_frequencies.count_nonzero() / self.codebook_size) * 100,
                'codebook/entropy': entropy(freq_np / freq_np.sum())
            }

            self.codebook_indices = []
            return codebook_dict
        else:
            return None
        
    def is_score_ready(self):
        return len(self.codebook_indices) == self.codebook_size