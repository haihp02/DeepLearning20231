import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def mlm_loss(output, target):
    if isinstance(output, dict):
        output = output['language_model_logits']
        loss = F.cross_entropy(output, target, ignore_index=-100)
    return loss

def constrastive_loss(output, tau=1):
    '''
    Loss for constrastive learning
    Score for positive sample always at first
    '''
    if isinstance(output, dict):
        scores = output['scores']
        exp_scores = torch.exp(scores/tau)
        loss = -torch.log(exp_scores[0]/exp_scores.sum())
    return loss