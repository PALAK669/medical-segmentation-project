import torch


class Trainer:

    def __init__(self, model, dataloader, optimizer, loss_fn, device):

        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self):

        self.model.train()

        total_loss = 0

        for images, masks in self.dataloader:

            images = images.to(self.device)
            masks = masks.to(self.device)

            preds = self.model(images)

            loss = self.loss_fn(preds, masks)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)