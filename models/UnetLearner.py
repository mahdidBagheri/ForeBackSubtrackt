import numpy as np
import torch.cuda
from tqdm import tqdm
from collections import OrderedDict

class Learner():
    def __init__(self, model, loss, optimizer, train_loader, test_loader):
        self.loss = loss
        self.model = model
        self.optimizer = optimizer
        self. train_loader = train_loader
        self.test_loader = test_loader

    def run_epoch(self,epoch, val=False):
        if not val:
            pbar = tqdm(self.train_loader, desc=f"train epoch {epoch}")
            self.model.train()
        else:
            pbar = tqdm(self.test_loader, desc=f"val epoch {epoch}")
            self.model.eval()

        outputs = []
        runing_loss = 0
        running_acc = 0
        for i , batch in enumerate(pbar):
            if not val:
                output = self.train_step(batch)
            else:
                output = self.test_step(batch)

            runing_loss += output["loss"]
            running_acc += output["acc"]
            output["runing_loss"] = (runing_loss/(i+1))
            output["running_acc"] = (running_acc/(i+1))

            pbar.set_postfix(output)
            outputs.append(output)
        self.schedule_lr()
        if not val:
            result = self.train_end(outputs)
        else:
            result = self.test_end(outputs)

        return result

    def step(self):
        self.optimizer.step()

    def train_step(self,batch):
        loss, acc = self.run_batch(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.step()
        output = OrderedDict({'loss': abs(loss.item()), 'acc': acc.item()})
        return output

    def train_end(self,outputs):
        loss_sum = 0
        for od in outputs:
            loss_sum += od["loss"]
        return loss_sum/len(outputs)

    def test_step(self,batch):
        loss, acc = self.run_batch(batch, val=True)
        output = OrderedDict({'loss': abs(loss.item()),'acc': acc})
        return output

    def test_end(self,outputs):
        loss_sum = 0
        for od in outputs:
            loss_sum += od["loss"]
        return loss_sum/len(outputs)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def schedule_lr(self):
        pass

    def run_batch(self, batch, val=False):
        input = batch[0]
        target = batch[1]
        if(torch.cuda.is_available()):
            input = input.cuda()
            target = target.cuda()
        output = self.model(input)
        acc = self.calc_accuracy(output,target)
        loss = self.loss(output,target)
        return loss, acc

    def calc_accuracy(self,output, target):
        diff = abs(output-target)
        acc = torch.sum(diff) / (output.size()[0]*output.size()[1]*output.size()[2]*output.size()[3])
        return acc