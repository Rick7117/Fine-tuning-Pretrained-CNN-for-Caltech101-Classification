import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import yaml
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from models.resnet import build_resnet_model

def evaluate_model(model, dataloader, dataset_size, class_names, model_type):
    print(f"Evaluating {model_type} model...")
    model.eval()  # Set model to evaluate mode

    running_corrects = 0
    all_labels = []
    all_preds = []

    # Iterate over data.
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(model.fc.weight.device)
            labels = labels.to(model.fc.weight.device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # statistics
            running_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_acc = running_corrects.double() / dataset_size

    print(f'{model_type} Accuracy: {epoch_acc:.4f}')

    # Calculate and print additional metrics
    # all_labels = np.array(all_labels)
    # all_preds = np.array(all_preds)

    # Confusion Matrix
    # cm = confusion_matrix(all_labels, all_preds)
    # print("Confusion Matrix:")
    # print(cm)

    # Classification Report
    # print("\nClassification Report:")
    # print(classification_report(all_labels, all_preds, target_names=class_names))

    # F1 Score (macro average)
    # f1 = f1_score(all_labels, all_preds, average='macro')
    # print(f"\nF1 Score (Macro Average): {f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned and scratch models.')
    parser.add_argument('--finetuned_model_path', type=str, required=True, help='Path to the fine-tuned model weights.')
    parser.add_argument('--scratch_model_path', type=str, required=True, help='Path to the scratch model weights.')
    args = parser.parse_args()

    # Load configuration from config.yaml
    with open('conf/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Define data transformations for evaluation set (using validation transforms)
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Caltech-101 train and validation datasets
    data_dir = 'data'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir):
        print(f"Error: Training data directory not found at {train_dir}")
        print("Please ensure the dataset is downloaded and organized.")
        exit()
    if not os.path.exists(val_dir):
        print(f"Error: Validation data directory not found at {val_dir}")
        print("Please ensure the dataset is downloaded and organized.")
        exit()

    image_dataset_train = datasets.ImageFolder(train_dir, data_transforms)
    image_dataset_val = datasets.ImageFolder(val_dir, data_transforms)

    # Combine datasets
    combined_dataset = torch.utils.data.ConcatDataset([image_dataset_train, image_dataset_val])
    dataloader_combined = DataLoader(combined_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4) # No need to shuffle for evaluation
    dataset_size_combined = len(combined_dataset)
    class_names = image_dataset_train.classes # Class names should be the same for both train and val

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Evaluate Fine-tuned Model ---
    model_ft = build_resnet_model(num_classes=len(class_names), pretrained=True) # Load architecture
    model_ft = model_ft.to(device)

    # Construct the path to the saved fine-tuned model weights
    model_ft_save_path = args.finetuned_model_path

    if os.path.exists(model_ft_save_path):
        model_ft.load_state_dict(torch.load(model_ft_save_path, map_location=device)) # Load weights
        evaluate_model(model_ft, dataloader_combined, dataset_size_combined, class_names, model_type='Fine-tuned')
    else:
        print(f"Fine-tuned model weights not found at {model_ft_save_path}. Please train the model first.")

    print("-" * 30)

    # --- Evaluate Model Trained from Scratch ---
    model_scratch = build_resnet_model(num_classes=len(class_names), pretrained=False) # Load architecture
    model_scratch = model_scratch.to(device)

    # Construct the path to the saved scratch model weights
    model_scratch_save_path = args.scratch_model_path

    if os.path.exists(model_scratch_save_path):
        model_scratch.load_state_dict(torch.load(model_scratch_save_path, map_location=device)) # Load weights
        evaluate_model(model_scratch, dataloader_combined, dataset_size_combined, class_names, model_type='Scratch')
    else:
        print(f"Scratch model weights not found at {model_scratch_save_path}. Please train the model first.")

    print("\nEvaluation complete!")