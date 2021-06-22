import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl

class MyModel(pl.LightningModule):

    def __init__(self, input_size=784, hidden_layers=None, output_size=10):
        super().__init__()
        self.save_hyperparameters()

        if hidden_layers is None:
            hidden_layers = [128, 64]

        # attach 
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
    
        tensorboard_logs = {'train_loss': loss}           # I mostly, think something like this should be mentioned
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.SGD(model.parameters(), lr=0.003)
   
   
    
    

if __name__ == '__main__':
    import argparse
    from pytorch_lightning import Trainer, seed_everything

    seed_everything(0)

    # data
    mnist_train = MNIST("./data/", train=True, download=True, transform=transforms.ToTensor())
    mnist_train = DataLoader(mnist_train, batch_size=32, num_workers=4)
    mnist_val = MNIST("./data/", train=True, download=True, transform=transforms.ToTensor())
    mnist_val = DataLoader(mnist_val, batch_size=32, num_workers=4)

    # model
    model = MyModel()

    # most basic trainer, uses good defaults
    trainer = Trainer(progress_bar_refresh_rate=20, max_epochs=10)
    trainer.fit(model, mnist_train, mnist_val)
    
    # export a snapshot of the trained model 
    trainer.save_checkpoint("./my_trained_model.ckpt")

