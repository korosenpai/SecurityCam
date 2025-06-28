import os
import cv2
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader

from constants import *

from dataset import ViolenceDataset
from LSTM_model import ViolenceLSTM, load_violence_dataset
from gradcam import apply_gradcam, predict_and_visualize



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
demo_path = "dataset/Violence/V_3.mp4"
predict_and_visualize(demo_path, test_ds, model, device)
