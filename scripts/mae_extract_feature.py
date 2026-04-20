import os
import numpy as np
import torch
import argparse
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


# ==========================================================
# Dense Window Builder (CORE FIX)
# ==========================================================
def build_dense_windows(frames, window_size=16):
    """
    frames: list of PIL Images
    returns: list of windows (each window = list of 16 frames)
    guarantees: len(windows) == len(frames)
    """
    T = len(frames)
    if T == 0:
        return []

    if T < window_size:
        frames = frames + [frames[-1]] * (window_size - T)
        T = len(frames)

    left = window_size // 2 - 1   # 7
    right = window_size - left - 1  # 8

    padded = (
        [frames[0]] * left +
        frames +
        [frames[-1]] * right
    )

    windows = []
    for t in range(T):
        window = padded[t : t + window_size]
        windows.append(window)

    return windows


# ==========================================================
# Argument Parser
# ==========================================================
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_root', required=True)
    parser.add_argument('--video_root', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--model_name', default='MCG-NJU/videomae-large')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--mode', nargs='+', type=str)
    parser.add_argument('--nth_layer', type=int, default=-1)
    parser.add_argument('--cache_dir', default=None)
    return parser


# ==========================================================
# Iterator (FIXED + UNIFIED)
# ==========================================================
def get_iterator(args, mode):
    batch_size = args.batch_size

    data = np.load(
        os.path.join(args.anno_root, f'{mode}_info.npy'),
        allow_pickle=True
    ).item()

    num = len(data) - 1
    ds_name = osp.split(args.anno_root)[-1]

    reader = VideoMAEFeatureReader(
        args.model_name,
        device=args.device,
        nth_layer=args.nth_layer,
        cache_dir=args.cache_dir
    )

    def iterate():
        for vid_idx in range(num):   # ← FIX: no shadowing
            fname = data[vid_idx]['folder']

            # ==================================================
            # Case 1: Phoenix / CSL (image folders)
            # ==================================================
            if ds_name in ['Phoenix14T', 'CSL-Daily']:
                image_paths = get_img_list(ds_name, args.video_root, fname)

                frames = [
                    Image.open(p).convert('RGB') for p in image_paths
                ]

            # ==================================================
            # Case 2: How2Sign (video stream)
            # ==================================================
            else:
                start_time = None

                if ds_name == 'How2Sign':
                    start_time = data[vid_idx]['original_info']['START_REALIGNED']
                    end_time = data[vid_idx]['original_info']['END_REALIGNED']

                    frames = read_video(
                        fname,
                        start_time=start_time,
                        end_time=end_time
                    )
                else:
                    continue

                if len(frames) == 0:
                    yield [], data[vid_idx]['fileid'], str(start_time)
                    continue

            # ==================================================
            # Dense Extraction (CORE LOGIC)
            # ==================================================
            windows = build_dense_windows(frames, window_size=16)

            video_feats = []

            for j in range(0, len(windows), batch_size):
                batch_windows = windows[j : j + batch_size]

                feats = reader.get_feats(batch_windows).cpu().numpy()
                video_feats.append(feats)

            video_feats = np.concatenate(video_feats, axis=0)  # (T, 1024)

            # ==================================================
            # Output
            # ==================================================
            if ds_name == 'How2Sign':
                yield video_feats, data[vid_idx]['fileid'], str(start_time)
            else:
                yield video_feats, data[vid_idx]['fileid'], None

    return iterate, num


# ==========================================================
# Main
# ==========================================================
def main():
    parser = get_parser()
    args = parser.parse_args()

    modes = ["dev", "test", "train"]

    for m in modes:
        ds_name = osp.split(args.anno_root)[-1]
        fname = f'mae_dense_feat_{ds_name}'

        save_path = osp.join(args.save_dir, fname, m)
        os.makedirs(save_path, exist_ok=True)

        if ds_name == 'How2Sign':
            _m = 'val' if m == 'dev' else m
        elif ds_name == 'NIASL2021':
            _m = 'validation' if m == 'dev' else m
        else:
            _m = m

        generator, num = get_iterator(args, _m)
        iterator = generator()

        for feats, fileid, st in tqdm.tqdm(iterator, total=num):
            postfix = "_dense"

            if st is not None:
                postfix = f'_{st}{postfix}'

            np.save(
                osp.join(save_path, f'{fileid}{postfix}.npy'),
                feats
            )


if __name__ == "__main__":
    main()