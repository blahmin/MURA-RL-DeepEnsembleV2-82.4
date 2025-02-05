import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import sys
import time
from datetime import datetime, timedelta

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from deepmodel import EnhancedEnsembleRL

class MURADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.body_parts = []

        body_parts = [d for d in os.listdir(root_dir) if d.startswith('XR_')]
        print("\nDataset composition:")
        for body_part in body_parts:
            part_dir = os.path.join(root_dir, body_part)
            part_count = 0
            for root, _, files in os.walk(part_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(root, file))
                        label = 1 if 'positive' in root.lower() else 0
                        self.labels.append(label)
                        self.body_parts.append(body_part)
                        part_count += 1
            print(f"{body_part}: {part_count} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        body_part = self.body_parts[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, body_part

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch_num):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    part_correct = {}
    part_total = {}

    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch_num}')
    for inputs, labels, body_parts in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs, body_parts[0])
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i, part in enumerate(body_parts):
            if part not in part_correct:
                part_correct[part] = 0
                part_total[part] = 0
            part_total[part] += 1
            if predicted[i] == labels[i]:
                part_correct[part] += 1

        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%',
            'lr': f'{current_lr:.6f}'
        })

        del inputs, labels, outputs, loss
        torch.cuda.empty_cache()

    part_accuracy = {part: 100 * part_correct[part] / part_total[part] for part in part_total}
    return running_loss / len(train_loader), 100 * correct / total, part_accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    part_correct = {}
    part_total = {}

    with torch.no_grad():
        for inputs, labels, body_parts in tqdm(val_loader, desc='Validating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs, body_parts[0])
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i, part in enumerate(body_parts):
                if part not in part_correct:
                    part_correct[part] = 0
                    part_total[part] = 0
                part_total[part] += 1
                if predicted[i] == labels[i]:
                    part_correct[part] += 1

            del inputs, labels, outputs
            torch.cuda.empty_cache()

    part_accuracy = {part: 100 * part_correct[part] / part_total[part] for part in part_total}
    return running_loss / len(val_loader), 100 * correct / total, part_accuracy


def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 64
    epochs = 200
    learning_rate = 1e-4
    num_classes = 2

    train_dir = r"C:\Users\blahm\PycharmProjects\Work\CNNXRAY\data\train"
    val_dir = r"C:\Users\blahm\PycharmProjects\Work\CNNXRAY\data\valid"

    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomResizedCrop(160, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.RandomAffine(
            degrees=20,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
            shear=10
        ),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomEqualize(p=1.0)
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MURADataset(train_dir, transform=train_transform)
    val_dataset = MURADataset(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    model = EnhancedEnsembleRL(num_classes=num_classes, beta=0.5).to(device)

    pretrained_path = os.path.join(os.path.dirname(train_dir), 'enhanced_ensemble_best.pth')
    if os.path.exists(pretrained_path):
        print("Loading pretrained weights...")
        checkpoint = torch.load(pretrained_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("Pretrained weights loaded successfully.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100
    )

    best_val_acc = 0
    best_part_accs = None

    print("\nStarting training...")
    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss, train_acc, train_part_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch + 1
        )
        val_loss, val_acc, val_part_acc = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch [{epoch+1}/{epochs}] - Time: {epoch_time/60:.2f} minutes")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("\nPer-body-part validation accuracy:")
        for part in val_part_acc:
            print(f"{part}: {val_part_acc[part]:.2f}%")
            if best_part_accs and part in best_part_accs and val_part_acc[part] > best_part_accs[part]:
                print(f"New best for {part}! Previous: {best_part_accs[part]:.2f}%")

        # Save model checkpoint if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_part_accs = val_part_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_acc,
                'val_part_accuracy': val_part_acc
            }, os.path.join(os.path.dirname(train_dir), 'ultimate_ensemble_RAAAH.pth'))
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")

        elapsed_time = time.time() - start_time
        estimated_total = elapsed_time * (epochs / (epoch + 1))
        estimated_remaining = estimated_total - elapsed_time
        print(f"\nTime elapsed: {elapsed_time/3600:.2f} hours")
        print(f"Estimated time remaining: {estimated_remaining/3600:.2f} hours")
        print(f"Estimated completion: {(datetime.now() + timedelta(seconds=estimated_remaining)).strftime('%Y-%m-%d %H:%M:%S')}")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
