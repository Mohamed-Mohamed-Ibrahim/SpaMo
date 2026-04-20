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
    H, W, C = frame.shape
    return keypoints, scores, [W, H]

def process_video_sequential(video_path, tgt_dir, fileid, wholebody, overwrite=False):
    output_path = os.path.join(tgt_dir, str(fileid).replace('/', '_') + ".pkl")
    
    if os.path.exists(output_path) and not overwrite:
        return

    data = {"keypoints": [], "scores": []}

    # 1. Read MP4 Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    vid_data = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        vid_data.append(frame)
    cap.release()

    if len(vid_data) == 0:
        return

    # 2. Sequential GPU Processing (NO THREADS)
    results = []
    for frame in vid_data:
        results.append(process_frame(frame, wholebody))

    # 3. Extract and format the primary person
    for keypoints, scores, w_h in results:
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

    with open(output_path, 'wb') as file:
        pickle.dump(data, file)

def get_video_path(video_root, fileid):
    exts = ['.mp4', '.mov', '.avi', '.mkv']
    for ext in exts:
        cand = os.path.join(video_root, fileid + ext)
        if os.path.exists(cand):
            return cand
    cand = os.path.join(video_root, fileid)
    if os.path.exists(cand):
        return cand
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_file', required=True, help='Direct path to the .npy annotation file')
    parser.add_argument('--video_root', required=True, help='root folder containing raw .mp4 videos')
    parser.add_argument('--pose_root', required=True, help='where to save pose .pkl files')
    parser.add_argument('--mode_name', required=True, help='Name of the split (e.g., dev, train, test)')
    
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--backend", default="onnxruntime", choices=["opencv", "onnxruntime", "openvino"])
    parser.add_argument("--openpose_skeleton", action="store_true", help="use openpose format")
    parser.add_argument("--mode", default="lightweight", choices=["performance", "lightweight", "balanced"])
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    wholebody = Wholebody(
        to_openpose=args.openpose_skeleton,
        mode=args.mode,
        backend=args.backend,
        device=args.device
    )

    save_dir = os.path.join(args.pose_root, args.mode_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading NPY annotations from: {args.anno_file}")
    
    # Clean NPY loading
    data = np.load(args.anno_file, allow_pickle=True).item()
    items = [data[k] for k in sorted(data.keys(), key=lambda x: int(x) if str(x).isdigit() else x)] if isinstance(data, dict) else list(data)

    print(f"\nProcessing [{args.mode_name}] split...")
    for entry in tqdm(items, desc=f'[{args.mode_name}]'):
        fileid = entry.get('name') or entry.get('fileid') or entry.get('id')
        if fileid is None:
            continue

        video_path = get_video_path(args.video_root, fileid)
        if video_path is None:
            continue

        process_video_sequential(
            video_path=video_path,
            tgt_dir=save_dir,
            fileid=fileid,
            wholebody=wholebody,
            overwrite=args.overwrite
        )

if __name__ == "__main__":
    main()