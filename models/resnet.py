import torch
import torch.nn as nn
import torchvision.models as models

def build_resnet_model(num_classes, pretrained=True):
    """
    Builds a ResNet model, optionally pre-trained, and modifies the final layer.

    Args:
        num_classes (int): The number of output classes.
        pretrained (bool): If True, loads a pre-trained model.

    Returns:
        torch.nn.Module: The modified ResNet model.
    """
    # Load a pre-trained ResNet-18 model
    model = models.resnet18(pretrained=pretrained)

    # Get the number of features in the last layer
    num_ftrs = model.fc.in_features

    # Replace the last layer with a new one that has the correct number of output classes
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model