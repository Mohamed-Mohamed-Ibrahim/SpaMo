import os
import numpy as np
import torch
import argparse
import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEModel, VideoMAEImageProcessor
from peft import PeftModel
import os.path as osp
import sys
import torch.multiprocessing as mp

# --- PATH FIX ---------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.helpers import sliding_window_for_list, read_video, get_img_list

# --- GLOBAL SETTINGS --------------------------------------------------------
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
class VideoMAEFeatureReader(object):
    def __init__(self, model_name, device, overlap_size, nth_layer, cache_dir=None, lora_path=None):
        self.device = device
        self.overlap_size = overlap_size
        self.nth_layer = nth_layer

        self.image_processor = VideoMAEImageProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = VideoMAEModel.from_pretrained(model_name)

        # ── Load LoRA adapter (if provided) ──────────────────────────
        # Expects a PEFT adapter directory saved by finetune_encoders.py
        # e.g. --lora_path logs/encoder_finetune/lora_weights/mae_lora
        if lora_path is not None:
            if os.path.isdir(lora_path):
                print(f"[LoRA MAE] Loading adapter from: {lora_path}")
                self.model = PeftModel.from_pretrained(self.model, lora_path)
                self.model = self.model.merge_and_unload()
                print("[LoRA MAE] Adapter merged into base model (zero overhead).")
            else:
                raise FileNotFoundError(
                    f"[LoRA MAE] Expected a PEFT adapter directory, got: {lora_path}\n"
                    f"  Run finetune_encoders.py first to generate mae_lora/ directory."
                )

        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def get_feats(self, video_batch):
        inputs = self.image_processor(images=video_batch, return_tensors="pt")
        inputs = inputs.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = self.model(**inputs, output_hidden_states=True).hidden_states
        
        feats = outputs[self.nth_layer][:, 0]
        return feats


# ----------------------------------------------------------------------------
class VideoDataset(Dataset):
    def __init__(self, args, mode, rank=0, world_size=1):
        self.args = args
        self.mode = mode

        # Load annotation file
        full_data = np.load(
            osp.join(args.anno_root, f"{mode}_info.npy"),
            allow_pickle=True
        ).item()

        # --- FIX: Filter out non-integer keys (like 'prefix') ---
        # Only keep keys that are integers (the video entries)
        valid_keys = [k for k in full_data.keys() if isinstance(k, int)]
        all_keys = sorted(valid_keys)
        # --------------------------------------------------------

        total_len = len(all_keys)
        split_len = total_len // world_size
        
        start_idx = rank * split_len
        if rank == world_size - 1:
            end_idx = total_len 
        else:
            end_idx = (rank + 1) * split_len
            
        my_keys = all_keys[start_idx:end_idx]
        self.data = {k: full_data[k] for k in my_keys}

        self.num_videos = len(self.data)
        self.ds_name = osp.split(args.anno_root)[-1]

        if rank == 0:
            print(f"[VideoDataset] '{mode}' initialized with {self.num_videos} videos (Total pool: {total_len}).")

        self.resize_size = 256
        self.crop_size = 224

    def __len__(self):
        return self.num_videos

    def process_frame(self, img):
        img = img.resize((self.resize_size, self.resize_size), resample=Image.BILINEAR)
        left = (self.resize_size - self.crop_size) // 2
        top = (self.resize_size - self.crop_size) // 2
        right = left + self.crop_size
        bottom = top + self.crop_size
        img = img.crop((left, top, right, bottom))
        return img

    def __getitem__(self, idx):
        # We need to map the 0..len index to the actual keys in our split
        key = list(self.data.keys())[idx]
        entry = self.data[key]
        
        fname, fileid = entry["folder"], entry["fileid"]
        start_time_str = None
        videos = []

        if self.ds_name in ["Phoenix14T", "CSL-Daily"]:
            image_list = get_img_list(self.ds_name, self.args.video_root, fname)
            if len(image_list) < 16:
                image_list += [image_list[-1]] * (16 - len(image_list))
            clips = sliding_window_for_list(image_list, 16, self.args.overlap_size)

            for clip in clips:
                pil_frames = []
                for path in clip:
                    try:
                        img = Image.open(path).convert("RGB")
                        img = self.process_frame(img)
                        pil_frames.append(img.copy())
                        img.close()
                    except Exception as e:
                        print(f"Warning: Failed to load image {path}: {e}")
                        continue
                if len(pil_frames) > 0:
                    videos.append(pil_frames)

        elif self.ds_name == "How2Sign" or self.ds_name == "Phoenix14TCompressed":
            s_val = entry["original_info"]["START_REALIGNED"]
            e_val = entry["original_info"]["END_REALIGNED"]

            try: s = float(s_val)
            except (ValueError, TypeError): s = None

            try: e = float(e_val)
            except (ValueError, TypeError): e = None

            start_time_str = str(s) if s is not None else "None"
            frames = read_video(fname, start_time=s, end_time=e)

            if len(frames) == 0:
                return [], fileid, start_time_str

            if len(frames) < 16:
                frames += [frames[-1]] * (16 - len(frames))

            processed = []
            for f in frames:
                if isinstance(f, np.ndarray):
                    img = Image.fromarray(f).convert("RGB")
                elif isinstance(f, Image.Image):
                    img = f.convert("RGB")
                else:
                    raise TypeError(f"Unexpected frame type: {type(f)}")
                
                img = self.process_frame(img)
                processed.append(img)

            videos = sliding_window_for_list(processed, 16, self.args.overlap_size)

        else:
            raise NotImplementedError(f"Unknown dataset: {self.ds_name}")

        return videos, fileid, start_time_str


# ----------------------------------------------------------------------------
def custom_collate_fn(batch):
    return batch[0]
# ----------------------------------------------------------------------------


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_root', required=True)
    parser.add_argument('--video_root', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--model_name', default='MCG-NJU/videomae-large')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--overlap_size', type=int, default=8)
    parser.add_argument('--mode', nargs='+', type=str)
    parser.add_argument('--nth_layer', type=int, default=-1)
    parser.add_argument('--cache_dir', default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lora_path', default=None,
                        help='Path to LoRA adapter dir or checkpoint file')
    return parser


def run_extraction(rank, world_size, args):
    device = f'cuda:{rank}'
    if rank == 0:
        print(f"--- Spawning {world_size} processes. Rank {rank} on {device} ---")

    reader = VideoMAEFeatureReader(
        args.model_name,
        device,
        args.overlap_size,
        args.nth_layer,
        args.cache_dir,
        lora_path=args.lora_path,
    )

    modes = args.mode if isinstance(args.mode, list) else [args.mode]
    for m in modes:
        ds_name = osp.split(args.anno_root)[-1]
        out_folder = f"mae_feat_{ds_name}"
        
        if ds_name == "How2Sign":    _m = "val" if m == "dev" else m
        elif ds_name == "NIASL2021": _m = "validation" if m == "dev" else m
        else:                        _m = m
        
        save_dir_split = osp.join(args.save_dir, out_folder, _m)
        if rank == 0:
            os.makedirs(save_dir_split, exist_ok=True)

        dataset = VideoDataset(args, _m, rank=rank, world_size=world_size)

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False, 
            collate_fn=custom_collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
      
        if rank == 0:
            print(f"Extracting '{_m}' to {save_dir_split}")
            iterator = tqdm.tqdm(dataloader, total=len(dataset))
        else:
            iterator = dataloader

        for videos, fileid, st in iterator:
            if not videos: 
                continue

            feats_per_video = []
            for j in range(0, len(videos), args.batch_size):
                chunk = videos[j : j + args.batch_size]
                feats = reader.get_feats(chunk).cpu().numpy()
                feats_per_video.append(feats)

            feats = np.concatenate(feats_per_video, axis=0)
            
            postfix = (f"_{st}" if st is not None else "") + f"_overlap-{args.overlap_size}"
            np.save(osp.join(save_dir_split, f"{fileid}{postfix}.npy"), feats)

    print(f"✅ Rank {rank} complete.")


def main():
    args = get_parser().parse_args()
    
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs.")
    
    if world_size > 1:
        mp.spawn(
            run_extraction,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        run_extraction(0, 1, args)

if __name__ == "__main__":
    main()
