from constants import *

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from captum.attr import LayerGradCam


def apply_gradcam(cnn_model, target_layer, input_tensor, target_class, device):
    # input_tensor must be [1, 3, 224, 224]
    gradcam = LayerGradCam(cnn_model, target_layer)

    # gradcam shenanigans to process heatmap
    input_tensor = input_tensor.to(device)
    attr = gradcam.attribute(input_tensor, target=target_class)
    attr = attr.squeeze().detach().cpu().numpy()  # (C, H, W) or (H, W)
    heatmap = np.maximum(attr, 0).mean(axis=0)   # average over channels
    heatmap = cv2.resize(heatmap, (input_tensor.shape[3], input_tensor.shape[2]))
    heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
    return heatmap

def predict_and_visualize(video_path, dataset, model, device):
    raw_frames, tensor_batch = dataset.read_video(video_path)
    feats = dataset.extract_features(tensor_batch).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(feats)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        idx = out.argmax(1).item()

    label = ["NonViolent", "Violent"][idx]
    print(f"Prediction: {label} with confidence {probs[idx]:.2f}")

    cnn_model = dataset.backbone.to(device).eval()
    target_layer = dataset.target_layer
    cv2.namedWindow("captum-visual", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("captum-visual", DISPLAY_SIZE[0] * 2, DISPLAY_SIZE[1])  # double width for side by side comp

    tensor_batch = tensor_batch.permute(1, 0, 2, 3) # (T, C, H, W) to match raw_frames

    for frame, img_tensor in zip(raw_frames, tensor_batch):
        try:
            # fixes some bugs we caused
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]
            elif img_tensor.dim() == 4:
                img_tensor = img_tensor.to(device)  # [1, 3, H, W] already
            else:
                raise ValueError(f"Unexpected tensor shape: {img_tensor.shape}")

            # transform images as training
            img_tensor = transforms.Resize((224, 224))(img_tensor)
            img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])(img_tensor)

            heatmap = apply_gradcam(cnn_model, target_layer, img_tensor, idx, device)
            heatmap_uint8 = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            heatmap_resized = cv2.resize(heatmap_color, (frame_bgr.shape[1], frame_bgr.shape[0])) # to match frame

            cv2.putText(frame_bgr, f"{label} ({probs[idx]:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # draw res on screen

            img_final = cv2.hconcat([frame_bgr, heatmap_resized])
            cv2.imshow("captum-visual", img_final)

            if cv2.waitKey(DELAY_MS) & 0xFF == ord('q'):
                break
        except Exception as e:
            print("something went wrong (not surprised :/ )", e)
            continue

    cv2.destroyAllWindows()
    return label

