import torch.nn.functional as F

def CrossEntropyLoss2d(inputs, targets, weight=None):
    n, c, h, w = inputs.size()
    log_p = F.log_softmax(inputs, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c) #(n*h*w, c)
    log_p = log_p[targets.view(n*h*w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = targets >= 0
    targets = targets[mask]
    loss = F.nll_loss(log_p, targets, weight=weight)
    return loss

