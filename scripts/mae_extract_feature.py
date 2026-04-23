import argparse
import os
import cv2
import numpy as np
import torch
import tqdm
import os.path as osp
from PIL import Image
from transformers import VideoMAEModel, VideoMAEImageProcessor

import sys
sys.path.append('./')

from utils.helpers import read_video, get_img_list

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True
# ==========================================================
# VideoMAE Feature Reader (UNCHANGED)
# ==========================================================
class VideoMAEFeatureReader(object):
    def __init__(
        self, 
        model_name='MCG-NJU/videomae-large', 
        cache_dir=None,
        device='cuda:0',
        nth_layer=-1
    ):
        self.device = device
        self.nth_layer = nth_layer

        self.image_processor = VideoMAEImageProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = VideoMAEModel.from_pretrained(model_name).to(self.device).eval()
        
    @torch.no_grad()
    def get_feats(self, video_batch):
        inputs = self.image_processor(images=video_batch, return_tensors="pt").to(self.device)
        
        outputs = self.model(**inputs, output_hidden_states=True).hidden_states
        outputs = outputs[self.nth_layer]          # (B, N_patches, 1024)
        outputs = outputs.mean(dim=1)             # (B, 1024)
        
        return outputs


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()
    return frames


def build_dense_windows(frames, window_size=16):
    T = len(frames)
    if T == 0:
        return []

    if T < window_size:
        frames = frames + [frames[-1]] * (window_size - T)
        T = len(frames)

    left = window_size // 2 - 1
    right = window_size - left - 1

    
    padded = [frames[0]] * left + frames + [frames[-1]] * right

    windows = []
    for t in range(T):
        windows.append(padded[t:t + window_size])

    return windows


def get_video_path(video_root, fileid, split_name):
    """
    Looks for the video inside video_root/split_name/ (e.g., videos/dev/)
    """
    exts = ['.mp4', '.mov', '.avi', '.mkv']
    
    # 1. Look inside the specific split folder (e.g., videos/dev/fileid.mp4)
    split_video_root = os.path.join(video_root, split_name)
    
    for ext in exts:
        cand = os.path.join(split_video_root, fileid + ext)
        if os.path.exists(cand):
            return cand
            
    # 2. Fallback: Look directly in the root just in case (e.g., videos/fileid.mp4)
    for ext in exts:
        cand = os.path.join(video_root, fileid + ext)
        if os.path.exists(cand):
            return cand
            
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_root', required=True)
    parser.add_argument('--video_root', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--model_name', default='MCG-NJU/videomae-large')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--nth_layer', type=int, default=-1)
    parser.add_argument('--cache_dir', default=None)

    args = parser.parse_args()

    reader = VideoMAEFeatureReader(
        model_name=args.model_name,
        device=args.device,
        nth_layer=args.nth_layer,
        cache_dir=args.cache_dir
    )

    for mode in ["dev", "test", "train"]:
        data = np.load(os.path.join(args.anno_root, f"{mode}_info.npy"), allow_pickle=True).item()
        num = len(data) - 1  # <-- KEPT AS REQUESTED

        save_path = os.path.join(args.save_dir, "motion_dense", mode)
        os.makedirs(save_path, exist_ok=True)

        for i in tqdm.tqdm(range(num), desc=mode):
            entry = data[i]
            fileid = entry.get('name') or entry.get('fileid') or entry.get('id')

            video_path = get_video_path(args.video_root, fileid, mode)
            if video_path is None:
                continue

            frames = read_video(video_path)
            if len(frames) == 0:
                continue

            windows = build_dense_windows(frames)

            feats = []
            for j in range(0, len(windows), args.batch_size):
                batch = windows[j:j+args.batch_size]
                f = reader.get_feats(batch).cpu().numpy()
                feats.append(f)

            feats = np.concatenate(feats, axis=0)
            np.save(os.path.join(save_path, f"{fileid}.npy"), feats)


if __name__ == "__main__":
    main()