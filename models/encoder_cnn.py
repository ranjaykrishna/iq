"""Genearates a representation for an image input.
"""

import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """Generates a representation for an image input.
    """

    def __init__(self, output_size):
        """Load the pretrained ResNet-152 and replace top fc layer.
        """
        super(EncoderCNN, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, output_size)
        self.bn = nn.BatchNorm1d(output_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights.
	"""
        self.cnn.fc.weight.data.normal_(0.0, 0.02)
        self.cnn.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract the image feature vectors.
	"""
        features = self.cnn(images)
        output = self.bn(features)
        return output
