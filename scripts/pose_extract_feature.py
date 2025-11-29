import argparse
import os
import os.path as osp
import glob
import sys
import numpy as np
import cv2
from tqdm import tqdm
import mediapipe as mp

def get_video_path(video_root, fileid):
    # 1. Check if it is a directory (Common for Phoenix14T: features/fullFrame-256x256px/train/name)
    # We search recursively or check specific patterns
    
    # Direct check
    cand = osp.join(video_root, fileid)
    if osp.exists(cand):
        return cand

    # Search for the folder/file
    for root, dirs, files in os.walk(video_root):
        # Check if fileid is a folder name here
        if fileid in dirs:
            return osp.join(root, fileid)
        # Check if fileid is a filename (minus extension)
        for f in files:
            if fileid in f:
                return osp.join(root, f)
                
    return None

def read_frames_from_path(path):
    """
    Generator that yields frames from either a video file or a folder of images.
    """
    if osp.isdir(path):
        # It's a folder of images (Phoenix14T standard)
        # Pattern usually: *.png
        images = sorted(glob.glob(osp.join(path, "*.png")))
        if not images:
            images = sorted(glob.glob(osp.join(path, "*.jpg")))
            
        for img_path in images:
            frame = cv2.imread(img_path)
            if frame is not None:
                yield frame
    else:
        # It's a video file
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()

def extract_pose_from_video(video_path, pose_model, kp_count=33, max_frames=None):
    frames_kps = []
    frame_idx = 0
    
    # Use the generator to handle both Folders and Video Files
    for frame in read_frames_from_path(video_path):
        if max_frames is not None and frame_idx >= max_frames:
            break

        # MediaPipe expects RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process using the PASSED model instance (much faster)
        res = pose_model.process(img_rgb)
        
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

    if len(frames_kps) == 0:
        return np.zeros((0, kp_count, 3), dtype=np.float32)
        
    # Return shape (T, 33, 3)
    return np.stack(frames_kps, axis=0)

def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--anno_root', required=True, help='root folder containing <mode>_info.npy annotations')
    p.add_argument('--video_root', required=True, help='root folder containing raw videos or image folders')
    p.add_argument('--pose_root', required=True, help='where to save pose .npy files')
    p.add_argument('--modes', nargs='+', default=['dev', 'test', 'train'], help='modes to process')
    p.add_argument('--kp_count', type=int, default=33)
    p.add_argument('--max_frames_per_video', type=int, default=None)
    p.add_argument('--skip_existing', action='store_true', help='skip if pose npy already exists')
    return p

def main():
    args = get_parser().parse_args()

    # --- OPTIMIZATION: Initialize Model ONCE outside the loop ---
    mp_pose = mp.solutions.pose
    pose_model = mp_pose.Pose(
        static_image_mode=False, 
        model_complexity=1, 
        enable_segmentation=False, 
        min_detection_confidence=0.5
    )
    # ------------------------------------------------------------

    for mode in args.modes:
        save_dir = osp.join(args.pose_root, mode)
        os.makedirs(save_dir, exist_ok=True)

        anno_path_ml = osp.join(args.anno_root, f'{mode}_info_ml.npy')
        anno_path = osp.join(args.anno_root, f'{mode}_info.npy')
        
        if osp.exists(anno_path_ml):
            data = np.load(anno_path_ml, allow_pickle=True).item()
        elif osp.exists(anno_path):
            data = np.load(anno_path, allow_pickle=True).item()
        else:
            print(f'Annotation file not found for mode {mode}. Skipping.')
            continue

        # Normalize data list
        items = []
        if isinstance(data, dict):
            try:
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
                continue

            save_path = osp.join(save_dir, f'{fileid}.npy')
            if args.skip_existing and osp.exists(save_path):
                continue

            video_path = get_video_path(args.video_root, fileid)
            if video_path is None:
                # Fallback: check if the 'folder' key exists in annotation (common in Phoenix)
                if isinstance(entry, dict) and 'folder' in entry:
                     # Attempt to construct path from 'folder' key
                     # Phoenix path usually: features/fullFrame-256x256px/<mode>/<folder_name>
                     fallback = osp.join(args.video_root, entry['folder'])
                     if osp.exists(fallback):
                         video_path = fallback

            if video_path is None:
                print(f'Video/Folder not found for fileid {fileid}. Skipping.')
                continue

            try:
                # Pass the pre-initialized model
                kps = extract_pose_from_video(
                    video_path, 
                    pose_model, 
                    kp_count=args.kp_count, 
                    max_frames=args.max_frames_per_video
                )
                np.save(save_path, kps.astype(np.float32))
            except Exception as e:
                print(f'Failed to process {fileid}:', e)

    # Clean up
    pose_model.close()

if __name__ == '__main__':
    main()