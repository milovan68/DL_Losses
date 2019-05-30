import torch
import torch.nn as nn
import torch.nn.functional as F

class SmoothLoss(nn.Module):
    def __init__(self):
        super().__init__()
       
    def forward(self,
                outputs,
                targets
                ):
        log_prob = nn.functional.log_softmax(outputs, dim=1)
        return -torch.sum(log_prob * targets)      

#Create smooth vectors from label vectors 
#for example in image 3-label classification vector [[1], [2]] transform to [[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
#this is preventing overfitting and generalize classification model
N = true_labels.size(0)
smoothed_labels = torch.full(size=(N, self.num_classes), fill_value=0.1 / (self.num_classes - 1)).cuda()
smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(true_labels, dim=1), value=0.9)