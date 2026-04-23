import argparse
import os
import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from transformers import AutoImageProcessor, CLIPVisionModel

import sys
sys.path.append('./')
from utils.s2wrapper import forward as multiscale_forward

class ViTFeatureReader:
    def __init__(self, model_name, device, scales, nth_layer, cache_dir):
        self.device = device
        self.scales = scales
        self.nth_layer = nth_layer

        self.model = CLIPVisionModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            cache_dir=cache_dir
        ).to(device).eval()

        self.processor = AutoImageProcessor.from_pretrained(model_name)

    # [RESTORED] Helper for the s2wrapper to call
    @torch.no_grad()
    def forward_features(self, inputs):
        outputs = self.model(inputs).hidden_states
        return outputs[self.nth_layer]

    @torch.no_grad()
    def get_feats(self, frames):
        inputs = self.processor(frames, return_tensors="pt").to(self.device).pixel_values
        
        # [EXPLICIT BYPASS]: If no scales are provided, skip the wrapper entirely.
        if not self.scales:
            # Runs standard Native CLIP (Outputs 256 patches per frame)
            outputs = self.forward_features(inputs)
        else:
            # Runs S2 Multi-Scale Wrapper (Outputs 1024 high-res patches per frame)
            outputs = multiscale_forward(
                self.forward_features, 
                inputs, 
                scales=self.scales, 
                num_prefix_token=1,
                resize_output_to_idx=1 
            )
        
        # Return the high-res patches, drop the [CLS] token
        return outputs[:, 0]


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


def get_video_path(video_root, fileid, split_name):
    exts = ['.mp4', '.mov', '.avi', '.mkv']
    split_video_root = os.path.join(video_root, split_name)
    
    for ext in exts:
        cand = os.path.join(split_video_root, fileid + ext)
        if os.path.exists(cand):
            return cand
            
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
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model_name', default='openai/clip-vit-large-patch14')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nth_layer', type=int, default=-1)
    parser.add_argument('--cache_dir', default=None)
    # [RESTORED] Correct scale factors for s2wrapper
    parser.add_argument('--scales', nargs='+', type=int, default= [])

    args = parser.parse_args()

    reader = ViTFeatureReader(
        args.model_name,
        args.device,
        args.scales,
        args.nth_layer,
        args.cache_dir
    )

    for mode in ["dev", "test", "train"]:
        data = np.load(os.path.join(args.anno_root, f"{mode}_info.npy"), allow_pickle=True).item()
        num = len(data) - 1

        save_path = os.path.join(args.save_dir, "spatial", mode)
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

            feats = []
            for j in range(0, len(frames), args.batch_size):
                batch = frames[j:j+args.batch_size]
                f = reader.get_feats(batch).cpu().numpy()
                feats.append(f)

            feats = np.concatenate(feats, axis=0).astype(np.float16)
            np.save(os.path.join(save_path, f"{fileid}.npy"), feats)


if __name__ == "__main__":
    main()