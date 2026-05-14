import argparse
import os
import os.path as osp
import tqdm
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, CLIPVisionModel
from peft import PeftModel
import torch.multiprocessing as mp

import sys, gc

import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Could not find image processor class")

# Get the absolute path to the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

print(f"Added to path: {parent_dir}")  # Debug line

from utils.s2wrapper import forward as multiscale_forward
from utils.helpers import read_video, get_img_list


_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)

torch.set_float32_matmul_precision("high")
NUM_WORKERS = os.cpu_count()


class FrameDataset(Dataset):
    """Handles parallel image loading and preprocessing on CPU workers."""

    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        item = self.image_paths[idx]
        if isinstance(item, str):
            img = Image.open(item).convert("RGB")
        else:
            img = item.convert("RGB")
        pixel_values = self.processor(img, return_tensors="pt").pixel_values.squeeze(0)
        return pixel_values


class ViTFeatureReader(object):
    def __init__(
        self,
        model_name="openai/clip-vit-large-patch14",
        cache_dir=None,
        device="cuda",
        s2_mode="s2wrapping",
        scales=[1, 2],
        nth_layer=-1,
        lora_path=None,
    ):
        self.s2_mode = s2_mode
        self.device = device
        self.scales = scales
        self.nth_layer = nth_layer

        self.model = (
            CLIPVisionModel.from_pretrained(
                model_name, output_hidden_states=True, cache_dir=cache_dir
            )
        )

        # ── Load LoRA adapter (if provided) ──────────────────────────
        # Expects a PEFT adapter directory saved by finetune_encoders.py
        # e.g. --lora_path logs/encoder_finetune/lora_weights/vit_lora
        if lora_path is not None:
            if os.path.isdir(lora_path):
                print(f"[LoRA ViT] Loading adapter from: {lora_path}")
                self.model = PeftModel.from_pretrained(self.model, lora_path)
                self.model = self.model.merge_and_unload()
                print("[LoRA ViT] Adapter merged into base model (zero overhead).")
            else:
                raise FileNotFoundError(
                    f"[LoRA ViT] Expected a PEFT adapter directory, got: {lora_path}\n"
                    f"  Run finetune_encoders.py first to generate vit_lora/ directory."
                )

        self.model = self.model.to(device).eval()

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model)

        # Optimization: PyTorch 2.0+ Graph Compilation (Speed boost after first batch)
        try:
            self.model = torch.compile(self.model)
        except Exception:
            print("Torch compile not supported; skipping.")

        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def forward_features(self, inputs):
        outputs = self.model(inputs).hidden_states[self.nth_layer]
        return outputs

    @torch.no_grad()
    def extract_features(self, pixel_values):
        with torch.amp.autocast('cuda'):
            if self.s2_mode == "s2wrapping":
                outputs = multiscale_forward(
                    self.forward_features,
                    pixel_values,
                    scales=self.scales,
                    num_prefix_token=1,
                )
            else:
                outputs = self._forward_logic(pixel_values)

            # Return only the [CLS] token (index 0)
            return outputs[:, 0].half().cpu().numpy()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_root", help="location of tsv files", required=True)
    parser.add_argument("--video_root", help="location of tsv files", required=True)
    parser.add_argument("--device", help="device to use", default="cuda:0")
    parser.add_argument("--s2_mode", default="")
    parser.add_argument(
        "--scales", nargs="+", type=int, help="List of scales", default=[]
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--nth_layer", type=int, default=-1)
    parser.add_argument("--cache_dir", help="cache dir for model", default=None)

    parser.add_argument("--save_dir", help="where to save the output", required=True)
    parser.add_argument(
        "--model_name", help="ViT model name", default="openai/clip-vit-large-patch14"
    )
    parser.add_argument(
        "--lora_path", help="Path to LoRA adapter dir or checkpoint file", default=None
    )

    return parser


def get_iterator(args, mode):
    data = np.load(
        os.path.join(args.anno_root, f"{mode}_info.npy"), allow_pickle=True
    ).item()
    num = len(data) - 1
    ds_name = osp.split(args.anno_root)[-1]
    reader = ViTFeatureReader(
        args.model_name,
        device=args.device,
        s2_mode=args.s2_mode,
        scales=args.scales,
        nth_layer=args.nth_layer,
        cache_dir=args.cache_dir,
        lora_path=args.lora_path,
    )

    def iterate():
        for i in range(num):
            fname = data[i]["folder"]
            file_id = data[i]["fileid"]
            start_time = data[i].get("original_info", {}).get("START_REALIGNED", None)

            if ds_name in ["Phoenix14T", "CSL-Daily"]:
                image_list = get_img_list(ds_name, args.video_root, fname)
            else:
                if ds_name == "How2Sign" or ds_name == "Phoenix14TCompressed":
                    image_list = read_video(
                        fname,
                        start_time=start_time,
                        end_time=data[i]["original_info"].get("END_REALIGNED"),
                    )

            if not image_list:
                yield [], file_id, str(start_time)
                continue

            dataset = FrameDataset(image_list, reader.image_processor)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=0,
                pin_memory=True,
                prefetch_factor=None,
            )

            video_feats = []
            for batch in loader:
                batch = batch.to(args.device, non_blocking=True)
                feats = reader.extract_features(batch)
                video_feats.append(feats)

            final_feats = np.concatenate(video_feats, axis=0), file_id, str(start_time)
            yield final_feats

            del video_feats
            del final_feats
            del loader
            del dataset
            if "cuda" in args.device:
                torch.cuda.empty_cache()
            gc.collect()

    return iterate, num


def main():
    mode = ["train", "dev", "test"]
    for m in mode:
        parser = get_parser()
        args = parser.parse_args()

        ds_name = osp.split(args.anno_root)[-1]
        _model_name = os.path.split(args.model_name)[-1]
        fname = f"{_model_name}_feat_{ds_name}"

        os.makedirs(osp.join(args.save_dir, fname, m), exist_ok=True)

        if ds_name == "How2Sign":
            if m == "dev":
                _m = "val"
            else:
                _m = m
        elif ds_name == "NIASL2021":
            if m == "dev":
                _m = "validation"
        else:
            _m = m

        generator, num = get_iterator(args, _m)
        iterator = generator()

        for vit_feat in tqdm.tqdm(iterator, total=num):
            feats, id, st = vit_feat
            save_path = osp.join(args.save_dir, fname, m)

            postfix = ""
            if args.s2_mode != "":
                postfix = f"_{args.s2_mode}"
            if len(args.scales) == 3:
                postfix = f"{postfix}_large"
            if st is not None:
                postfix = f"_{st}{postfix}"

            np.save(osp.join(save_path, f"{id}{postfix}.npy"), feats)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
