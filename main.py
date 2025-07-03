import json
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



# Load data
X_train, X_test, y_train, y_test = load_violence_dataset()
train_ds = ViolenceDataset(X_train, y_train)
test_ds = ViolenceDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViolenceLSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_losses = []
train_accuracies = []

skip_training = os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH)

# if trained model file not found, train
if not skip_training:
    print("Training model...")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nepoch {epoch}/{NUM_EPOCHS}")

        model.train()
        tot_loss, correct, total = 0.0, 0, 0

        for batch_idx, (raw, features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features) # [batch_size, num_classes]
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            preds = outputs.argmax(1) # take index of class with higher score for each row
            batch_correct = (preds == labels).sum().item()
            batch_total = labels.size(0)
            accuracy = batch_correct / batch_total

            # to measure accuracy
            tot_loss += batch_loss * batch_total
            correct += batch_correct
            total += batch_total

        epoch_loss = tot_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"epoch {epoch} done: loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.2%}")

    # save model and metrics
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"model saved to {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump({"loss": train_losses, "accuracy": train_accuracies}, f)
    print(f"metrics saved to {METRICS_PATH}")


else:
    print("model file found, not training")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
        train_losses = metrics["loss"]
        train_accuracies = metrics["accuracy"]

# Plot training history
'''
if train_losses and train_accuracies:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
    plt.title('training loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', color='green')
    plt.title('training accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Evaluation
print("Evaluating on test set...")
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for _, features, labels in test_loader:
        features = features.to(device)
        output = model(features)
        pred = output.argmax(1).cpu().tolist()
        y_pred.extend(pred)
        y_true.extend(labels.tolist())

print(classification_report(y_true, y_pred, target_names=['NonViolent', 'Violent']))
'''


print("Running demo on a sample video with Grad-CAM visualization...")
correctSum = 0

for file in os.listdir("dataset/NonViolence"):
    demo_path = os.path.join("dataset/NonViolence", file)
    try:
        x = predict_and_visualize(demo_path, test_ds, model, device)
        if x == "Violent":
            correctSum += 1
    except Exception as e:
        print(f"error on file: {demo_path} : {e}")

for file in os.listdir("dataset/NonViolence"):
    demo_path = os.path.join("dataset/NonViolence", file)
    try:
        x = predict_and_visualize(demo_path, test_ds, model, device)
        if x == "NonViolent":
            correctSum += 1
    except Exception as e:
        print(f"error on file: {demo_path} : {e}")

denominator = len(os.listdir("dataset/Violence")) + len(os.listdir("dataset/NonViolence"))
print(f"Accuracy is: {correctSum / denominator:.2%}")

