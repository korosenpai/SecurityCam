from constants import *

import os
import torch.nn as nn
from sklearn.model_selection import train_test_split

class ViolenceLSTM(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# load dataset
def load_violence_dataset(base_path=DATA_PATH):
    video_paths, labels = [], []
    for cls, val in [('Violence', 1), ('NonViolence', 0)]:
        folder = os.path.join(base_path, cls)
        for fname in os.listdir(folder):
            if fname.lower().endswith('.mp4'):
                video_paths.append(os.path.join(folder, fname))
                labels.append(val)
                
    return train_test_split(video_paths, labels, test_size=0.2, stratify=labels, random_state=42)
