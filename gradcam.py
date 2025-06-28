from constants import *

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from captum.attr import LayerGradCam

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
