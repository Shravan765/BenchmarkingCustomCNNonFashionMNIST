import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import torch

torch.set_num_threads(8)

from collections import defaultdict
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2
from tqdm import tqdm

from MyCNNModel import (
    MyCNNModel_v1_0,
    MyCNNModel_v1_1,
    MyCNNModel_v2_0,
    MyCNNModel_v2_1,
    MyCNNModel_v3,
    MyCNNModel_v4,
    MyCNNModel_v5
)

# -----------------------------------------------------------------------------
# Seeding Function
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Dataset & DataLoader Setup
# -----------------------------------------------------------------------------
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,))
])

train_transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5), # horizontally inverted fashion article is still the same
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,)),
])

train_dataset = FashionMNIST(root='./dataset', train=True, download=False, transform=train_transforms)
test_dataset = FashionMNIST(root='./dataset', train=False, download=False, transform=transform)

train_size = int(9/10 * len(train_dataset))
val_size = int(1/10 * len(train_dataset))

g = torch.Generator().manual_seed(SEED)

train_sub , val_sub = random_split(train_dataset, [train_size, val_size], generator=g)
train_loader = DataLoader(train_sub, batch_size=64, shuffle=True, generator=g, num_workers=2)
validation_loader = DataLoader(val_sub, batch_size=64, shuffle=False, generator=g, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, generator=g, num_workers=2)

# -----------------------------------------------------------------------------
# Experiment Setup & Data Structures
# -----------------------------------------------------------------------------
ITER_COUNT = 5
epochs_list = [10]
model_classes = [
    MyCNNModel_v1_0,
    MyCNNModel_v1_1,
    MyCNNModel_v2_0,
    MyCNNModel_v2_1,
    MyCNNModel_v3,
    MyCNNModel_v4,
    MyCNNModel_v5
]

# Stores list of accuracies: key = (model_name, num_epochs) -> list of floats
results = defaultdict(list)

# -----------------------------------------------------------------------------
# Iteration & Training Loop
# -----------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()

def eval_loop(model, loader):
    correct = 0
    total = 0
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validation"):
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            pred = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    
    avg_test_loss = test_loss / len(loader)
    accuracy = correct / total
    return avg_test_loss, correct, total, accuracy

for iteration in range(1, ITER_COUNT + 1):
    print(f"\n{'='*80}\n STARTING EXPERIMENT ITERATION {iteration}/{ITER_COUNT} \n{'='*80}")
    
    # Re-seed with an iteration offset to ensure different, reproducible seeds per run
    set_seed(SEED + iteration)

    for num_epochs in epochs_list:
        for model_cls in model_classes:
            model = model_cls().to(device)
            model_name = model.__class__.__name__

            model = torch.compile(model)
            
            print("#" * 80, "\n")
            print(f"{'!'*10}\nIter: {iteration}/{ITER_COUNT} | Epochs: {num_epochs} | Model: {model_name}\n{'!'*10}\n")
        
            optim = torch.optim.Adam(model.parameters())

            # --- Training Loop ---
            for epoch in range(num_epochs):
                train_correct = 0
                train_total = 0
                model.train()
                running_loss = 0.0
                for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                    imgs, labels = imgs.to(device), labels.to(device)
                    train_total += labels.size(0)
                    optim.zero_grad()
                    outputs = model(imgs)

                    pred = torch.argmax(outputs, 1)
                    train_correct += (pred == labels).sum().item()

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optim.step()
                    
                    running_loss += loss.item()
                
                print(f'Training Epoch [{epoch+1}/{num_epochs}], \
                      Average Loss: {running_loss / len(train_loader):.3f}, \
                        Correct/Total = {train_correct}/{train_total}')

                # --- Evaluation (Validation) Loop ---
                print("\n!!!!! Validation !!!!!")

                avg_test_loss, correct, total, accuracy = eval_loop(model, validation_loader)
                
                print(f"Average Test Loss : {avg_test_loss:.3f}")
                print(f"Correct = {correct} , Total = {total}")
                print(f"Accuracy = {accuracy:.4f}\n")


            # --- Evaluation (Test) Loop ---
            print("\n!!!!! Evaluation !!!!!")

            avg_test_loss, correct, total, accuracy = eval_loop(model, test_loader)
            
            # Record final accuracy for this iteration
            results[(model_name, num_epochs)].append(accuracy)
            
            print(f"Average Test Loss : {avg_test_loss:.3f}")
            print(f"Correct = {correct} , Total = {total}")
            print(f"Accuracy = {accuracy:.4f}\n")

# -----------------------------------------------------------------------------
# Summary Results across all 5 iterations
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print(f"{'SUMMARY RESULTS (ACROSS ' + str(ITER_COUNT) + ' ITERATIONS)':^80}")
print("="*80)
print(f"{'Model Name':<20} | {'Epochs':<8} | {'Mean Accuracy':<15} | {'Std Dev':<15}")
print("-" * 65)

for (model_name, num_epochs), acc_list in results.items():
    mean_acc = np.mean(acc_list)
    std_acc = np.std(acc_list, ddof=1)  # Sample standard deviation
    print(f"{model_name:<20} | {num_epochs:<8} | {mean_acc:.4f} ({(mean_acc*100):.2f}%) | {std_acc:.4f}")
print("="*80)