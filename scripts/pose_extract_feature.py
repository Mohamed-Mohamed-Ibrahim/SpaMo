


import argparse
import os
import os.path as osp
import sys
import numpy as np
import cv2
from tqdm import tqdm
import mediapipe as mp # for pose extraction


def get_video_path(video_root, fileid):
    # try common extensions
    exts = ['.mp4', '.mov', '.avi', '.mkv']
    for ext in exts:
        cand = osp.join(video_root, fileid + ext)
        if osp.exists(cand):
            return cand
    # maybe fileid already is a path relative to video_root
    cand = osp.join(video_root, fileid)
    if osp.exists(cand):
        return cand
    # try searching for fileid as substring in video_root
    for root, _, files in os.walk(video_root):
        for f in files:
            if fileid in f:
                return osp.join(root, f)
    return None


def extract_pose_from_video(video_path, mp, kp_count=33, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f'Cannot open video: {video_path}')

    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
    frames_kps = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # MediaPipe expects RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)
        if res.pose_landmarks:
            kps = []
            for lm in res.pose_landmarks.landmark:
                # normalized coordinates x,y in [0,1], z is relative depth
                kps.append([lm.x, lm.y, lm.z])
            kps = np.array(kps, dtype=np.float32)
        else:
            kps = np.zeros((kp_count, 3), dtype=np.float32)
        frames_kps.append(kps)
        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break
    cap.release()
    pose.close()
    if len(frames_kps) == 0:
        return np.zeros((0, kp_count, 3), dtype=np.float32)
    return np.stack(frames_kps, axis=0)


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--anno_root', required=True, help='root folder containing <mode>_info.npy annotations')
    p.add_argument('--video_root', required=True, help='root folder containing raw videos')
    p.add_argument('--pose_root', required=True, help='where to save pose .npy files (feature-root; will create subfolders per mode)')
    p.add_argument('--modes', nargs='+', default=['dev', 'test', 'train'], help='modes to process (names without _info.npy suffix)')
    p.add_argument('--kp_count', type=int, default=33)
    p.add_argument('--max_frames_per_video', type=int, default=None)
    p.add_argument('--skip_existing', action='store_true', help='skip if pose npy already exists')
    return p


def main():
    args = get_parser().parse_args()
    # follow other feature scripts style: create a subfolder per mode inside pose_root
    for mode in args.modes:
        save_dir = osp.join(args.pose_root, mode)
        os.makedirs(save_dir, exist_ok=True)

        # try multilanguage annotation first, then fallback
        anno_path_ml = osp.join(args.anno_root, f'{mode}_info_ml.npy')
        anno_path = osp.join(args.anno_root, f'{mode}_info.npy')
        if osp.exists(anno_path_ml):
            data = np.load(anno_path_ml, allow_pickle=True).item()
        elif osp.exists(anno_path):
            data = np.load(anno_path, allow_pickle=True).item()
        else:
            print(f'Annotation file not found for mode {mode}: checked {anno_path_ml} and {anno_path}. Skipping.')
            continue

        # normalize data into a list of entries (like other scripts expect)
        items = []
        if isinstance(data, dict):
            # some preprocessed files are dicts keyed by index
            # keep ordering stable
            try:
                # keys might be numeric strings
                keys = sorted(data.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
            except Exception:
                keys = sorted(data.keys())
            for k in keys:
                items.append(data[k])
        else:
            items = list(data)

        for entry in tqdm(items, desc=f'[{mode}]'):
            fileid = None
            if isinstance(entry, dict):
                fileid = entry.get('fileid') or entry.get('id')
            elif hasattr(entry, 'fileid'):
                fileid = entry.fileid

            if fileid is None:
                print('Entry missing fileid, skipping:', entry)
                continue

            save_path = osp.join(save_dir, f'{fileid}.npy')
            if args.skip_existing and osp.exists(save_path):
                continue

            video_path = get_video_path(args.video_root, fileid)
            if video_path is None:
                print(f'Video not found for fileid {fileid}. Skipping.')
                continue

            try:
                kps = extract_pose_from_video(video_path, mp, kp_count=args.kp_count, max_frames=args.max_frames_per_video)
                np.save(save_path, kps.astype(np.float32))
            except Exception as e:
                print(f'Failed to process {fileid} ({video_path}):', e)


if __name__ == '__main__':
    main()

