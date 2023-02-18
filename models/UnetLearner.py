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
            pbar = tqdm(self.train_loader, desc=f"epoch {epoch}")
            self.model.train()
        else:
            pbar = tqdm(self.test_loader, desc=f"epoch {epoch}")
            self.model.eval()

        outputs = []
        for i , batch in enumerate(pbar):
            if not val:
                output = self.train_step(batch)
            else:
                output = self.test_step(batch)

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
        loss = self.run_batch(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.step()
        output = OrderedDict({'loss': abs(loss.item())})
        return output

    def train_end(self,outputs):
        pass

    def test_step(self,batch):
        pass

    def test_end(self,outputs):
        pass

    def save(self):
        pass

    def schedule_lr(self):
        pass

    def run_batch(self, batch):
        input = batch[0]
        target = batch[1]
        output = self.model(input)
        loss = self.loss(output,target)
        return loss