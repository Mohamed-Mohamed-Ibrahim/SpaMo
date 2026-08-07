"""
Emotion Feature Extraction Pipeline
===========================================
Extracts facial emotion embeddings from video datasets using Haar cascades 
and a fine-tuned ViT-B/16 model.

"""

import os
import argparse
import warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from transformers import ViTModel, AutoImageProcessor, logging as hf_logging
from scipy.interpolate import interp1d

# Suppress non-critical warnings
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

MODEL_NAME = "trpakov/vit-face-expression"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".webm")


def load_models():
    print(f"[INFO] Loading ViT emotion encoder  : {MODEL_NAME}")
    fe = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=False)
    model = ViTModel.from_pretrained(MODEL_NAME)
    
    model.eval().to(DEVICE)
    
    # ---------------------------------------------------------
    # MULTI-GPU DYNAMIC ALLOCATION
    # ---------------------------------------------------------
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"[INFO] Detected {num_gpus} GPUs! Wrapping model in DataParallel.")
        model = nn.DataParallel(model)
    else:
        print(f"[INFO] Model loaded on {DEVICE} (1 GPU or CPU).")
        
    print(f"[INFO] Loading Haar cascade face detector")
    detector = cv2.CascadeClassifier(HAAR_PATH)
    if detector.empty():
        raise RuntimeError(f"Haar cascade not found at: {HAAR_PATH}")
    print(f"[INFO] Face detector ready\n")

    return fe, model, detector


def load_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def downsample(frames: list, stride: int):
    indices = list(range(0, len(frames), stride))
    sampled = [frames[i] for i in indices]
    return sampled, indices


def detect_face(frame_bgr: np.ndarray, detector: cv2.CascadeClassifier):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    roi = cv2.cvtColor(frame_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    return Image.fromarray(roi)


@torch.no_grad()
def get_embeddings_batch(faces: list, feature_extractor, model) -> list:
    """Processes a batch of face images through the ViT model."""
    inputs = feature_extractor(images=faces, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    outputs = model(**inputs)
    
    # Extract the CLS token (index 0) from the last hidden state for the entire batch
    cls_vecs = outputs.last_hidden_state[:, 0, :]
    
    return list(cls_vecs.cpu().numpy())


def interpolate_missing(raw_embeddings: list, valid_mask: list) -> np.ndarray:
    F = len(raw_embeddings)
    valid_idx = [i for i, v in enumerate(valid_mask) if v]
    valid_embs = np.stack([raw_embeddings[i] for i in valid_idx])
    hidden_size = valid_embs.shape[1]

    if len(valid_idx) == F:
        return valid_embs.astype(np.float32)

    Ze = np.zeros((F, hidden_size), dtype=np.float32)
    all_idx = np.arange(F)

    for dim in range(hidden_size):
        fn = interp1d(
            valid_idx, valid_embs[:, dim],
            kind="linear", bounds_error=False,
            fill_value=(valid_embs[0, dim], valid_embs[-1, dim])
        )
        Ze[:, dim] = fn(all_idx)

    return Ze


def extract_ze(video_path: str, fe, model, detector, stride: int, batch_size: int) -> np.ndarray:
    frames = load_frames(video_path)
    if not frames:
        raise RuntimeError("Empty video.")

    sampled, _ = downsample(frames, stride)
    
    valid_faces = []
    valid_mask = []

    # 1. Detect faces for all sampled frames
    for frame in sampled:
        face = detect_face(frame, detector)
        if face is not None:
            valid_faces.append(face)
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    if not valid_faces:
        raise RuntimeError("No face detected in any frame.")

    # 2. Extract embeddings in batches (enables multi-GPU)
    valid_embs = []
    for i in range(0, len(valid_faces), batch_size):
        batch = valid_faces[i:i + batch_size]
        batch_embs = get_embeddings_batch(batch, fe, model)
        valid_embs.extend(batch_embs)

    # 3. Reconstruct the sequence with None for missing faces
    raw_embeddings = [None] * len(sampled)
    emb_idx = 0
    for i, is_valid in enumerate(valid_mask):
        if is_valid:
            raw_embeddings[i] = valid_embs[emb_idx]
            emb_idx += 1

    # 4. Interpolate to fill gaps
    return interpolate_missing(raw_embeddings, valid_mask)


def process_dataset(input_dir: str, output_dir: str, stride: int, batch_size: int):
    os.makedirs(output_dir, exist_ok=True)
    fe, model, detector = load_models()

    video_tasks = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(VIDEO_EXTS):
                full_path = os.path.join(root, fname)
                rel_dir = os.path.relpath(root, input_dir)
                video_tasks.append((full_path, rel_dir, fname))

    total = len(video_tasks)
    print(f"[INFO] Total videos found: {total}\n")

    done = 0
    failed = []

    for video_path, rel_dir, fname in video_tasks:
        done += 1
        video_name = os.path.splitext(fname)[0]
        
        save_dir = os.path.join(output_dir, rel_dir) if rel_dir != "." else output_dir
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{video_name}_Ze.npy")

        if os.path.exists(save_path):
            print(f"[SKIP] ({done}/{total}) {rel_dir}/{video_name}")
            continue

        print(f"[{done}/{total}] {rel_dir}/{video_name}", end="  ")
        try:
            Ze = extract_ze(video_path, fe, model, detector, stride, batch_size)
            np.save(save_path, Ze)
            print(f"Ze{Ze.shape} ✓")
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append((video_path, str(e)))

    print("\n" + "="*60)
    print(f"[DONE] {done - len(failed)}/{total} videos processed successfully.")
    if failed:
        print(f"[WARN] {len(failed)} failed:")
        for path, err in failed:
            print(f"       {path}: {err}")
    print(f"[INFO] Features saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU Emotion Embedding Extractor")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Root directory containing video files")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save the extracted .npy features")
    parser.add_argument("-s", "--stride", type=int, default=8, help="Temporal downsampling stride (default: 8)")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for ViT inference (default: 32)")
    
    args = parser.parse_args()

    print("="*60)
    print("Multi-GPU Emotion Feature Extractor")
    print("="*60)
    
    process_dataset(args.input_dir, args.output_dir, args.stride, args.batch_size)