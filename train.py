# Basic structure for training a CNN on Caltech-101

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import time
import copy
from torch.utils.tensorboard import SummaryWriter
from models.resnet import build_resnet_model
import yaml
import argparse

# Define training function
def train_model(model, criterion, optimizer, scheduler, num_epochs, model_type='', config=None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Initialize TensorBoard writer
    # Create a unique log directory name based on hyperparameters
    if model_type == 'scratch':
        log_dir = f"runs/{model_type}_epochs{num_epochs}_bs{config['training']['batch_size']}_lr_scratch{config['optimizer']['scratch']['lr']}"
    else:
        
        log_dir = f"runs/{model_type}_epochs{num_epochs}_bs{config['training']['batch_size']}_lr_ft_new{config['optimizer']['finetune']['lr_new_layers']}_lr_ft_pre{config['optimizer']['finetune']['lr_pretrained_layers']}"
    writer = SummaryWriter(log_dir)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Log metrics to TensorBoard
            if phase == 'train':
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)
            else:
                writer.add_scalar('Loss/val', epoch_loss, epoch)
                writer.add_scalar('Accuracy/val', epoch_acc, epoch)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    # Save best model weights
    # Save best model weights with hyperparameters in filename
    if model_type == 'scratch':
        model_save_path = f"models/best_model_weights_{model_type}_epochs{num_epochs}_bs{config['training']['batch_size']}_lr_scratch{config['optimizer']['scratch']['lr']}"
    else:
        model_save_path = f"models/best_model_weights_{model_type}_epochs{num_epochs}_bs{config['training']['batch_size']}_lr_ft_new{config['optimizer']['finetune']['lr_new_layers']}_lr_ft_pre{config['optimizer']['finetune']['lr_pretrained_layers']}.pth"
    torch.save(model.state_dict(), model_save_path)
    writer.close()
    return model

# Main execution block
if __name__ == '__main__':
    # Load configuration from config.yaml
    parser = argparse.ArgumentParser(description='Train a CNN on Caltech-101.')
    parser.add_argument('--config', type=str, default='conf/config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load Caltech-101 dataset
    # The dataset needs to be downloaded and organized into 'train' and 'val' folders
    # with subfolders for each class.
    # Example structure:
    # data/
    #   train/
    #     class1/
    #     class2/
    #     ...
    #   val/
    #     class1/
    #     class2/
    #     ...

    data_dir = 'data' 

    # Check if data directory exists and contains train/val folders
    if not os.path.exists(os.path.join(data_dir, 'train')) or not os.path.exists(os.path.join(data_dir, 'val')):
        print("Error: Caltech-101 dataset not found in the expected structure.")
        print("Please download the dataset and organize it into 'train' and 'val' folders.")
        print("You can download the dataset from: https://data.caltech.edu/records/mzrjq-6wc02")
        exit()

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load pre-trained model (e.g., ResNet-18) using the function from models.resnet
    model_ft = build_resnet_model(num_classes=len(class_names), pretrained=True)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Implement fine-tuning with different learning rates
    # Parameters of newly constructed modules have requires_grad=True by default
    # Exclude fc parameters from the pretrained layers group
    pretrained_params = []
    for name, param in model_ft.named_parameters():
        if 'fc' not in name:
            pretrained_params.append(param)

    optimizer_ft = optim.SGD([
        {'params': model_ft.fc.parameters(), 'lr': config['optimizer']['finetune']['lr_new_layers']}, # New layer with higher LR
        {'params': pretrained_params, 'lr': config['optimizer']['finetune']['lr_pretrained_layers']} # Pre-trained layers with lower LR
    ], momentum=config['optimizer']['finetune']['momentum'])

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler_ft = optim.lr_scheduler.StepLR(optimizer_ft, step_size=config['scheduler']['step_size'], gamma=config['scheduler']['gamma'])

    print("\nFine-tuning pre-trained model:")
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler_ft, num_epochs=config['training']['num_epochs'], model_type='finetuned', config=config)

    # Implement training from scratch for comparison using the function from models.resnet
    model_scratch = build_resnet_model(num_classes=len(class_names), pretrained=False)
    model_scratch = model_scratch.to(device)
    optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=config['optimizer']['scratch']['lr'], momentum=config['optimizer']['scratch']['momentum']) # Higher LR for training from scratch
    exp_lr_scheduler_scratch = optim.lr_scheduler.StepLR(optimizer_scratch, step_size=config['scheduler']['step_size'], gamma=config['scheduler']['gamma'])

    print("\nTraining model from scratch:")
    model_scratch = train_model(model_scratch, criterion, optimizer_scratch, exp_lr_scheduler_scratch, num_epochs=config['training']['num_epochs'], model_type='scratch', config=config)

    print("Done!")