import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Dataset with GPU‐batch transforms and antialiasing ---
class ViolenceDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=16, size=(112, 112), device='cuda'):
        print("Initializing ViolenceDataset...")
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.size = size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
        self.cnn.eval()
        for p in self.cnn.parameters():
            p.requires_grad = False

        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])
        print(f"  Loaded {len(self.video_paths)} videos.")

    def __len__(self):
        return len(self.video_paths)

    def read_video(self, path, idx=None):
        if idx is not None:
            print(f"[Dataset] Reading video {idx}: {path}")
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.size)
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError(f"Could not read any frames from {path}")
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        batch = np.stack(frames, axis=0)
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2) / 255.0
        return tensor  # (T,3,H,W)

    def extract_features(self, frame_batch, idx=None):
        if idx is not None:
            print(f"[Dataset] Extracting features for video {idx}")
        frame_batch = frame_batch.to(self.device)
        frame_batch = self.transform(frame_batch)
        with torch.no_grad():
            feats = self.cnn(frame_batch)
            feats = feats.view(feats.size(0), -1)
        return feats

    def __getitem__(self, idx):
        frames = self.read_video(self.video_paths[idx], idx)
        feats = self.extract_features(frames, idx)
        return feats.cpu(), torch.tensor(self.labels[idx], dtype=torch.long)

# --- Load Data Utility ---
def load_violence_dataset(base_path='dataset'):
    print("Loading dataset from:", base_path)
    video_paths, labels = [], []
    for cls, val in [('Violence',1), ('NonViolence',0)]:
        folder = os.path.join(base_path, cls)
        for f in os.listdir(folder):
            if f.endswith('.mp4'):
                video_paths.append(os.path.join(folder,f))
                labels.append(val)
    X_train, X_test, y_train, y_test = train_test_split(
        video_paths, labels, test_size=0.2, stratify=labels, random_state=42)
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

# --- Model Definition ---
class ViolenceLSTM(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=False)
        self.fc   = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# --- Single-Video Prediction ---
def predict_single(video_path, dataset, model, device):
    print(f"\nPredicting on single video: {video_path}")
    frames = dataset.read_video(video_path)
    feats  = dataset.extract_features(frames)
    feats  = feats.unsqueeze(0).to(device)  # (1, T, 512)
    with torch.no_grad():
        out     = model(feats)
        prob    = F.softmax(out, dim=1).cpu().numpy()[0]
        pred_id = out.argmax(1).item()
    label_str = ["NonViolent","Violent"][pred_id]
    print(f"Prediction: {label_str} (nonviolent={prob[0]:.3f}, violent={prob[1]:.3f})")
    return label_str, prob

# --- Main Script ---
if __name__ == "__main__":
    # Paths & hyperparams
    model_path = 'violence_lstm_model.pth'
    num_epochs = 5

    # 1. Prepare data & dataset objects
    X_train, X_test, y_train, y_test = load_violence_dataset('dataset')
    train_ds = ViolenceDataset(X_train, y_train, device='cuda')
    test_ds  = ViolenceDataset(X_test,  y_test,  device='cuda')
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=2, pin_memory=True)

    # 2. Initialize model, optimizer, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model     = ViolenceLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 3. Load or train
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}, loading and skipping training.")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("No saved model found, starting training...")
        train_losses, train_accs = [], []
        for epoch in range(1, num_epochs+1):
            print(f"\n=== Epoch {epoch}/{num_epochs} ===")
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            for batch_idx, (feats, labels) in enumerate(train_loader, 1):
                print(f"[Training] Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}")
                feats, labels = feats.to(device), labels.to(device)
                outputs = model(feats)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * feats.size(0)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
            avg_loss = total_loss / total
            acc      = correct / total
            train_losses.append(avg_loss)
            train_accs.append(acc)
            print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

        # Save model
        print(f"\nSaving trained model to {model_path}")
        torch.save(model.state_dict(), model_path)

        # Plot training curves
        print("Plotting training curves...")
        epochs = range(1, num_epochs+1)
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(epochs, train_losses, marker='o')
        plt.title('Training Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True)
        plt.subplot(1,2,2)
        plt.plot(epochs, train_accs, marker='o')
        plt.title('Training Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 4. Evaluate on test set
        print("\nEvaluating on test set...")
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_idx, (feats, labels) in enumerate(test_loader, 1):
                print(f"[Eval] Batch {batch_idx}/{len(test_loader)}")
                feats = feats.to(device)
                outs  = model(feats)
                preds = outs.argmax(1).cpu().tolist()
                y_pred.extend(preds)
                y_true.extend(labels.tolist())
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['NonViolent','Violent']))

    # 5. Single-video demo
    example_path = "dataset/Violence/V_456.mp4" 
    predict_single(example_path, test_ds, model, device)
