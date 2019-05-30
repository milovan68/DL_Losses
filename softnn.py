import torch
import torch.nn.functional as F
import numpy as np

class SoftNearestNeighbour(torch.nn.Module):

    def __init__(self, temperature=100.0):
        super(SoftNearestNeighbour, self).__init__()
        self.temperature= temperature

    def forward(self, x, y):
        mask = self.masked_pick_probability(x, y)
        summed_masked_pick_prob = torch.sum(mask, dim=1)
        return torch.mean(- torch.log(0.00001 + summed_masked_pick_prob))

    def masked_pick_probability(self, x, y):
        return self.pick_probability(x) * self.same_label_mask(y, y)

    def pick_probability(self, x):
        f = self.fits(x, x) - torch.eye(x.size(0)).to('cuda')
        return f / (0.00001 + torch.sum(f, dim=1).expand(x.size(0)))

    def same_label_mask(self, y, y2):
        return (torch.eq(y, torch.unsqueeze(y2, dim=1)).squeeze(2).float()).float()


    def fits(self, A, B):
        return torch.exp(-(self.cosine_sim(A, B)) / self.temperature)
    
    def cosine_sim(self, A, B):
        return 1 - torch.mm(F.normalize(A), torch.transpose(F.normalize(B), 1,0))
