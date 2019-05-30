#creating another dataset in pytorch like a train_dataset with random categories 
#on every iteration on batch do
images, _ = next(iter(dataloaders['ood']))

#create uniform vector such as [0.25, 0.25, 0.25, 0.25]
uniform_vector = torch.full(size=(images.size(0), self.num_classes), fill_value=1 / self.num_classes).cuda()
logsoftmax = nn.LogSoftmax()
loss1_u = torch.mean(torch.sum(- uniform_vector * logsoftmax(out_u), 1))

#add this loss to main classification loss, coefficients before loss1_u tune from task to task
loss = classification_loss + 0.01 * (loss1_u)

#for inference use temperature in logits (temperature range [1.4-2.5]) and threshold with scores in softmax output 
score = F.softmax(a / temperature, dim=1)

