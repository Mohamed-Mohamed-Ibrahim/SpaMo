import argparse
import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from rtmlib import Wholebody


def process_frame(frame, wholebody):
    frame = np.uint8(frame)
    keypoints, scores = wholebody(frame)
    H, W, _ = frame.shape
    return keypoints, scores, [W, H]


def process_video_sequential(video_path, tgt_dir, fileid, wholebody, overwrite=False):
    output_path = os.path.join(tgt_dir, str(fileid).replace('/', '_') + ".pkl")

    if os.path.exists(output_path) and not overwrite:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        return

    data = {"keypoints": [], "scores": []}

    for frame in frames:
        keypoints, scores, w_h = process_frame(frame, wholebody)

        if len(keypoints) > 0:
            person_idx = np.argmax(scores.mean(axis=1)) if scores.ndim > 1 else 0
            kp = keypoints[person_idx] / np.array(w_h)
            sc = scores[person_idx]
        else:
            kp = np.zeros((133, 2), dtype=np.float32)
            sc = np.zeros((133,), dtype=np.float32)

        data['keypoints'].append(kp)
        data['scores'].append(sc)

    data['keypoints'] = np.array(data['keypoints'], dtype=np.float32)
    data['scores'] = np.array(data['scores'], dtype=np.float32)

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


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
    parser.add_argument('--pose_root', required=True)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default="onnxruntime")
    parser.add_argument("--mode", default="lightweight")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    wholebody = Wholebody(
        mode=args.mode,
        backend=args.backend,
        device=args.device
    )

    for mode_name in ["dev", "test", "train"]:
        anno_file = os.path.join(args.anno_root, f"{mode_name}_info.npy")

        if not os.path.exists(anno_file):
            continue

        save_dir = os.path.join(args.pose_root, mode_name)
        os.makedirs(save_dir, exist_ok=True)

        data = np.load(anno_file, allow_pickle=True).item()
        items = list(data.values()) if isinstance(data, dict) else list(data)

        num = len(data) - 1 

        for i in tqdm(range(num), desc=mode_name):
            entry = data[i]
            fileid = entry.get('name') or entry.get('fileid') or entry.get('id')
            if fileid is None:
                continue

            video_path = get_video_path(args.video_root, fileid, mode_name)
            if video_path is None:
                continue

            process_video_sequential(video_path, save_dir, fileid, wholebody, args.overwrite)


if __name__ == "__main__":
    main()