from dataset.p14t import Phoenix14T

print("🕵️ Loading training dataset...")

# Initializing exactly as your YAML config specifies
dataset = Phoenix14T(
    anno_root="./preprocess/Phoenix14T",
    vid_root="./dataset/Phoenix14T",
    feat_root="feat_Phoenix14T",
    mae_feat_root="mae_feat_Phoenix14T",
    pose_root="pose_Phoenix14T",
    mode="train",
    spatial=True,
    spatiotemporal=True,
    pose=False,
    spatial_postfix="",
    spatiotemporal_postfix="",
    pose_postfix=""
)

total = len(dataset)
print(f"✅ Loaded {total} items. Starting sequential processing...")

# Process items one by one to find the infinite loop
for i in range(total):
    # Print the index BEFORE processing so we know exactly where it hangs
    print(f"Processing index {i} / {total}...", end="\r")
    
    try:
        _ = dataset[i]
    except Exception as e:
        print(f"\n❌ Crashed on index {i} with error: {e}")

print("\n✅ All items processed without looping!")