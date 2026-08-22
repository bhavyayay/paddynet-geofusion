import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.segmentation.dataset import PaddyPatchDataset
from src.segmentation.model import PatchClassifierCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "models/patch_classifier_final.pt"


def train_one_fold(train_indices, val_index, dataset, num_classes=3):
    train_subset = Subset(dataset, train_indices)
    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)

    model = PatchClassifierCNN(in_channels=7, num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for patches, labels in train_loader:
            patches, labels = patches.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

    model.eval()
    val_patch, val_label = dataset[val_index]
    val_patch = val_patch.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        val_output = model(val_patch)
        predicted_label = torch.argmax(val_output, dim=1).item()

    return predicted_label, val_label.item(), model


def run_leave_one_out_cv():
    dataset = PaddyPatchDataset()
    n_samples = len(dataset)

    all_indices = list(range(n_samples))
    correct = 0
    predictions = []
    true_labels = []

    print(f"\nRunning Leave-One-Out Cross-Validation on {n_samples} samples...\n")

    for val_index in all_indices:
        train_indices = [i for i in all_indices if i != val_index]
        predicted_label, true_label, _ = train_one_fold(train_indices, val_index, dataset)

        is_correct = predicted_label == true_label
        correct += int(is_correct)
        predictions.append(predicted_label)
        true_labels.append(true_label)

        status = "CORRECT" if is_correct else "WRONG"
        print(f"Fold {val_index + 1}/{n_samples} -> predicted={predicted_label}, actual={true_label} [{status}]")

    accuracy = correct / n_samples
    print(f"\nLeave-One-Out Cross-Validation Accuracy: {accuracy:.2%} ({correct}/{n_samples})")

    return accuracy, predictions, true_labels


def train_final_model_on_all_data():
    dataset = PaddyPatchDataset()
    all_indices = list(range(len(dataset)))

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = PatchClassifierCNN(in_channels=7, num_classes=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining final model on all available data...")
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for patches, labels in train_loader:
            patches, labels = patches.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} -> loss: {epoch_loss:.4f}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nFinal model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    accuracy, predictions, true_labels = run_leave_one_out_cv()
    train_final_model_on_all_data()

    print("\n--- Summary ---")
    print(f"Leave-One-Out CV Accuracy: {accuracy:.2%}")
    print("Note: With only 12 samples, this accuracy is illustrative, not statistically robust.")
    print("Real ground-truth GPS survey data (100+ points) is required before this number is meaningful.")