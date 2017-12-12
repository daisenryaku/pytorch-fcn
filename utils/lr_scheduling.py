def lr_scheduler(optimizer, init_lr, iter_num, lr_decay_iter=1, max_iter=30000, power=0.9):
    if iter_num % lr_decay_iter or iter > max_iter:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * (1 - iter_num/max_iter)**power
