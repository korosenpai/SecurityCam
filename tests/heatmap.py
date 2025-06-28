import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from captum.attr import LayerGradCam

# === CONFIGURABLE PARAMETERS ===
NUM_FRAMES   = 32
FRAME_SIZE   = (112, 112)
DISPLAY_SIZE = (800, 600)
DELAY_MS     = 300
BATCH_SIZE   = 2
NUM_EPOCHS   = 20
MODEL_PATH   = '../trained_models/violence_lstm_model.pth'
DATA_PATH    = '../dataset'

# --- Dataset ---
class ViolenceDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=NUM_FRAMES, size=FRAME_SIZE, device='cuda'):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.size = size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load pretrained ResNet18
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        self.backbone = backbone
        self.cnn = nn.Sequential(*list(backbone.children())[:-1]).to(self.device)
        self.target_layer = backbone.layer4[-1]
        self.cnn.eval()
        for p in self.cnn.parameters():
            p.requires_grad = False

        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.video_paths)

    def read_video(self, path):
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
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {path}")
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        raw_batch = frames
        np_batch = np.stack(frames, axis=0)
        tensor = torch.from_numpy(np_batch).permute(0, 3, 1, 2) / 255.0
        return raw_batch, tensor

    def extract_features(self, tensor_batch):
        tensor_batch = tensor_batch.to(self.device)
        tensor_batch = self.transform(tensor_batch)
        with torch.no_grad():
            feats = self.cnn(tensor_batch)
            feats = feats.view(feats.size(0), -1)
        return feats

    def __getitem__(self, idx):
        raw_frames, tensor_batch = self.read_video(self.video_paths[idx])
        feats = self.extract_features(tensor_batch)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return raw_frames, feats.cpu(), label

# --- LSTM Model ---
class ViolenceLSTM(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# --- Load dataset ---
def load_violence_dataset(base_path=DATA_PATH):
    video_paths, labels = [], []
    for cls, val in [('Violence', 1), ('NonViolence', 0)]:
        folder = os.path.join(base_path, cls)
        for fname in os.listdir(folder):
            if fname.lower().endswith('.mp4'):
                video_paths.append(os.path.join(folder, fname))
                labels.append(val)
    return train_test_split(video_paths, labels, test_size=0.2, stratify=labels, random_state=42)

# --- GradCAM utility ---
def apply_gradcam(cnn_model, target_layer, input_tensor, target_class, device):
    input_tensor = input_tensor.unsqueeze(0).to(device)
    gradcam = LayerGradCam(cnn_model, target_layer)
    attr = gradcam.attribute(input_tensor, target=target_class)
    attr = attr.squeeze().detach().cpu().numpy()  
    heatmap = np.maximum(attr, 0).mean(axis=0)
    heatmap = cv2.resize(heatmap, (input_tensor.shape[3], input_tensor.shape[2]))
    heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
    return heatmap

# --- Visualization with GradCAM ---
def predict_and_visualize(video_path, dataset, model, device):
    raw_frames, tensor_batch = dataset.read_video(video_path)
    feats = dataset.extract_features(tensor_batch).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(feats)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        idx = out.argmax(1).item()

    label = ["NonViolent", "Violent"][idx]
    text = f"{label} ({probs[idx]:.2f})"
    print(f"Prediction: {label} with confidence {probs[idx]:.2f}")

    cnn_model = dataset.backbone.to(device).eval()
    target_layer = dataset.target_layer
    cv2.namedWindow("Analysis", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Analysis", *DISPLAY_SIZE)

    for frame, img_tensor in zip(raw_frames, tensor_batch):
        try:
            heatmap = apply_gradcam(cnn_model, target_layer, img_tensor, idx, device)
            heatmap_uint8 = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            overlayed = cv2.addWeighted(frame_bgr, 0.6, heatmap_color, 0.4, 0)
            cv2.putText(overlayed, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Analysis", overlayed)
            if cv2.waitKey(DELAY_MS) & 0xFF == ord('q'):
                break
        except Exception as e:
            print("Grad-CAM failed on a frame:", e)
            continue

    cv2.destroyAllWindows()

# --- Main ---
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_violence_dataset()
    train_ds = ViolenceDataset(X_train, y_train)
    test_ds = ViolenceDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViolenceLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print("Training model...")
        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
            model.train()
            tot_loss, correct, total = 0.0, 0, 0
            for batch_idx, (raw, feats, labels) in enumerate(train_loader):
                feats, labels = feats.to(device), labels.to(device)
                outputs = model(feats)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss = loss.item()
                preds = outputs.argmax(1)
                batch_correct = (preds == labels).sum().item()
                batch_total = labels.size(0)
                accuracy = batch_correct / batch_total
                tot_loss += batch_loss * batch_total
                correct += batch_correct
                total += batch_total
                print(f"[Epoch {epoch}/{NUM_EPOCHS} | Batch {batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {batch_loss:.4f} | Acc: {accuracy:.2%} | "
                      f"Device: {feats.device.type}")
            epoch_loss = tot_loss / total
            epoch_acc  = correct / total
            print(f"Epoch {epoch} Summary → Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2%}")
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        print("Evaluating on test set...")
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for _, feats, labels in test_loader:
                feats = feats.to(device)
                output = model(feats)
                pred = output.argmax(1).cpu().tolist()
                y_pred.extend(pred)
                y_true.extend(labels.tolist())
        print(classification_report(y_true, y_pred, target_names=['NonViolent', 'Violent']))

    print("Running demo on a sample video with Grad-CAM visualization...")
    demo_path = f"{DATA_PATH}/Violence/V_3.mp4"
    predict_and_visualize(demo_path, test_ds, model, device)

