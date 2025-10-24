import os
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# -----------------------
# Config & reproducibility
# -----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BATCH_SIZE = 64
LR = 0.001
EPOCHS = 10  # between 5 and 10 per assignment
OUTPUT_DIR = Path("output_fashion_mnist")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -----------------------
# Dataset and dataloaders
# -----------------------
transform = transforms.Compose([
    transforms.ToTensor(),  # scales to [0,1] and converts to torch.Tensor
])

train_dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# label names
CLASS_NAMES = train_dataset.classes  # ['T-shirt/top', 'Trouser', ...]


# -----------------------
# Model
# -----------------------
class SimpleMLP(nn.Module):
    def _init_(self, input_dim=784, h1=256, h2=128, num_classes=10):
        super()._init_()
        self.net = nn.Sequential(
            nn.Flatten(),  # 1x28x28 -> 784
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, num_classes)
            # CrossEntropyLoss applies softmax internally
        )

    def forward(self, x):
        return self.net(x)


model = SimpleMLP().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------
# Training & Evaluation helpers
# -----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return epoch_loss, epoch_acc, all_preds, all_targets


# -----------------------
# Training loop
# -----------------------
train_losses, train_accs = [], []
test_losses, test_accs = [], []

best_test_acc = 0.0
best_model_path = OUTPUT_DIR / "best_model.pth"

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, DEVICE)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"Epoch {epoch}/{EPOCHS} | Train loss: {train_loss:.4f} acc: {train_acc*100:.2f}% | "
          f"Test loss: {test_loss:.4f} acc: {test_acc*100:.2f}%")

    # save best
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
        }, best_model_path)

print(f"Best test accuracy: {best_test_acc*100:.2f}% (saved to {best_model_path})")

# -----------------------
# Final evaluation, confusion matrix, sample predictions
# -----------------------
test_loss, test_acc, all_preds, all_targets = evaluate(model, test_loader, criterion, DEVICE)
print(f"Final Test Loss: {test_loss:.4f} | Final Test Acc: {test_acc*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(all_targets, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
fig_cm, ax_cm = plt.subplots(figsize=(9, 9))
disp.plot(ax=ax_cm, xticks_rotation='vertical', cmap=plt.cm.Blues)
plt.title("Confusion Matrix (test set)")
plt.tight_layout()
cm_path = OUTPUT_DIR / "confusion_matrix.png"
fig_cm.savefig(cm_path)
print(f"Saved confusion matrix to {cm_path}")

# Classification report (console)
print("\nClassification report (test set):\n")
print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4))

# Save loss/accuracy curves
epochs_range = range(1, EPOCHS + 1)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(epochs_range, train_losses, marker='o', label='train loss')
axs[0].plot(epochs_range, test_losses, marker='o', label='test loss')
axs[0].set_title("Loss per epoch")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].legend()
axs[1].plot(epochs_range, train_accs, marker='o', label='train acc')
axs[1].plot(epochs_range, test_accs, marker='o', label='test acc')
axs[1].set_title("Accuracy per epoch")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Accuracy")
axs[1].legend()
plt.tight_layout()
plots_path = OUTPUT_DIR / "loss_acc_plots.png"
fig.savefig(plots_path)
print(f"Saved loss & accuracy plots to {plots_path}")

# Sample images with predictions: collect some correct and incorrect
model.eval()
correct_examples = []
incorrect_examples = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        outputs = model(inputs)
        _, preds = outputs.max(1)
        for i in range(inputs.size(0)):
            img = inputs[i].cpu().squeeze(0).numpy()  # 28x28
            true = targets[i].item()
            pred = preds[i].item()
            if true == pred and len(correct_examples) < 6:
                correct_examples.append((img, true, pred))
            if true != pred and len(incorrect_examples) < 6:
                incorrect_examples.append((img, true, pred))
        if len(correct_examples) >= 6 and len(incorrect_examples) >= 6:
            break

def plot_examples(examples, title, save_name):
    n = len(examples)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.0))
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.axis('off')
    for i, (img, true, pred) in enumerate(examples):
        ax = axes[i]
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(f"True: {CLASS_NAMES[true]}\nPred: {CLASS_NAMES[pred]}")
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUTPUT_DIR / save_name
    fig.savefig(path)
    print(f"Saved '{title}' to {path}")

plot_examples(correct_examples, "Correct Predictions (examples)", "correct_examples.png")
plot_examples(incorrect_examples, "Incorrect Predictions (examples)", "incorrect_examples.png")

# Save final model state (non-checkpoint)
final_model_path = OUTPUT_DIR / "final_model_state.pth"
torch.save(model.state_dict(), final_model_path)
print(f"Saved final model state_dict to {final_model_path}")

# Short conclusion (print and save)
conclusion = f"""
Conclusion:
- Model: Simple MLP (784 -> 256 -> 128 -> 10)
- Optimizer: Adam (lr={LR}), Loss: CrossEntropyLoss
- Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}
- Final Test Accuracy: {test_acc*100:.2f}%
- Best Test Accuracy: {best_test_acc*100:.2f}%

Observations:
- The MLP (fully-connected) achieves solid performance on Fashion-MNIST. If you need higher accuracy:
  1) consider simple CNNs (Conv2d -> ReLU -> Pool) which typically outperform MLPs on image tasks,
  2) add dropout / weight decay to reduce overfitting,
  3) tune learning rate or use a scheduler,
  4) apply mild normalization (mean/std) and data augmentation.

Files saved to: {OUTPUT_DIR.resolve()}
"""

print(conclusion)
with open(OUTPUT_DIR / "conclusion.txt", "w", encoding="utf-8") as f:
    f.write(conclusion)

# show figures on-screen if running interactively
try:
    plt.show()
except Exception:
    pass
