def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

#inputs, labels from train dataloader, calculate loss (criterion is loss from pytorch.nn module)
inputs, targets_a, targets_b, lam = self.mixup_data(inputs, labels, 1.0)
inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
#calculate accuracy on train dataset
running_corrects += (lam * preds.eq(targets_a.data).cpu().sum().float() + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float())
