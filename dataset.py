from constants import *

import os
import torch
import cv2
import torch.nn as nn
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights

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

