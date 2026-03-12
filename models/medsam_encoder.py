import torch
import torch.nn as nn


class MedSAMEncoder(nn.Module):

    def __init__(self, sam_model, frozen_layers=8):
        super().__init__()

        self.encoder = sam_model.image_encoder

        for i, block in enumerate(self.encoder.blocks):

            if i < frozen_layers:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, x):

        tokens = self.encoder(x)

        return tokens