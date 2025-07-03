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
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = max(total // self.num_frames, 1)
        count = 0
        while len(frames) < self.num_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % sample_rate == 0:
                frame = cv2.resize(frame, (112, 112))
                frame_rgb = frame[:, :, [2, 1, 0]]  # BGR to RGB
                frames.append(frame_rgb)
            count += 1
        cap.release()

        # pad with last frame if not enough
        while len(frames) < self.num_frames:
            frames.append(frames[-1])  # repeat last frame

        raw_frames = np.stack(frames).astype(np.uint8)  # (T, H, W, C)
        frames = raw_frames.astype(np.float32).transpose(3, 0, 1, 2) / 255.0  # (C, T, H, W)

        return raw_frames, torch.tensor(frames)


    def extract_features(self, tensor_batch):
    # tensor_batch: (C, T, H, W)
        T = tensor_batch.shape[1]
        frames = tensor_batch.permute(1, 0, 2, 3)  # → (T, C, H, W)

        transformed_frames = torch.stack([self.transform(frame) for frame in frames])  # (T, 3, 224, 224)

        transformed_frames = transformed_frames.to(self.device)
        with torch.no_grad():
            feats = self.cnn(transformed_frames)  # (T, 512, 1, 1)
            feats = feats.view(feats.size(0), -1)  # (T, 512)
        return feats

    def __getitem__(self, idx):
        raw_frames, tensor_batch = self.read_video(self.video_paths[idx])
        feats = self.extract_features(tensor_batch)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return raw_frames, feats.cpu(), label

