# Modular PRogramming

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm


class Trainer():
    '''Trainer Classe '''

    def __init__(self, model, device, train_loader, test_loader, optimizer, start_epoch, epoch, schedul, my_path, best_acc):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.start_epoch = int(start_epoch)
        self.epoch = int(epoch)
        self.scheduler = schedul
        self.path = my_path
        self.best_acc = best_acc

    def train(self, l1_lambda=0, l2_lambda=0):

        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        comb_train_losses = []
        comb_train_acc = []
        comb_test_losses = []
        comb_test_acc = []
        comb_lr = []
        comb_acc = []
        # EPOCHS = 10
        for epoch in range(self.start_epoch, self.epoch):

            print("EPOCH:", epoch)
            new_lr = self.optimizer.param_groups[0]['lr']
            print('learning rate ',
                  new_lr)

            # trainer = Trainer(model,device,train_loader,test_loader,optimizer,epoch)

            train_ac, train_los = self.train_mod()
            self.scheduler.step()
            test_ac, test_los = self.test_mod()
            # self.scheduler.step(test_los[-1])

            comb_train_losses.extend(train_los)
            comb_train_acc.extend(train_ac)
            comb_test_losses.extend(test_los)
            comb_test_acc.extend(test_ac)
            comb_lr.append(new_lr)
            comb_acc.append(train_ac[-1])

            if(test_ac[-1] > self.best_acc):

                print("...BEST MODEL...")
                best_ac = test_ac[-1]
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': test_los[-1],
                    'learn_rate_last': self.optimizer.param_groups[0]['lr']
                }, self.path)
                print("Save success...")

            if epoch % 10 == 0:
                torch.cuda.empty_cache()
                print('emptying the cache')

        return ((comb_train_acc, comb_train_losses), (comb_test_acc, comb_test_losses), (comb_lr, comb_acc))

    def train_mod(self):

        train_losses = []
        train_acc = []

        self.model.train()
        pbar = tqdm(self.train_loader)

        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(
                self.device), target.to(self.device)

            # Init
            self.optimizer.zero_grad()

            # Predict
            y_pred = self.model(data)

            # Calculate loss

            criterion = F.nll_loss(y_pred, target)

            # l1 regularization
            l1_reg_loss = sum(
                [torch.sum(abs(param)) for param in self.model.parameters()])

            # l2 regularization
            l2_reg_loss = sum([torch.sum(param**2)
                               for param in self.model.parameters()])

            # Calculate loss (depending on which decay parameter passed, regularization is calculated. )

            loss = criterion + self.l1_lambda * l1_reg_loss + \
                self.l2_lambda*l2_reg_loss

            train_losses.append(loss)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            # Update pbar-tqdm

            # get the index of the max log-probability
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)
                               ).sum().item()
            processed += len(data)

            pbar.set_description(
                desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            train_acc.append(100*correct/processed)
        return (train_acc, train_losses)

    def test_mod(self):
        test_losses = []
        test_acc = []
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(
                    self.device), target.to(self.device)
                output = self.model(data)
                # sum up batch loss
                test_loss += F.nll_loss(output,
                                        target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)
                                   ).sum().item()

        test_loss /= len(self.test_loader.dataset)
        test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(
                self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

        test_acc.append(100. * correct /
                        len(self.test_loader.dataset))

        return (test_acc, test_losses)
