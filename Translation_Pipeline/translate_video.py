import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import argparse

# Add parent directory to path so we can import from original repo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoImageProcessor, CLIPVisionModel
from transformers import VideoMAEModel, VideoMAEImageProcessor

from utils.s2wrapper import forward as multiscale_forward
from utils.helpers import sliding_window_for_list
import yaml

from spamo.t5_slt import FlanT5SLT

# Define constants
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
TARGET_W = 210
TARGET_H = 260

def get_args():
    parser = argparse.ArgumentParser()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_video = os.path.join(script_dir, 'translation_target')
    default_ckpt = os.path.join(script_dir, '..', 'weights', 'spamo.ckpt')
    default_cfg = os.path.join(script_dir, '..', 'configs', 'finetune.yaml')
    
    parser.add_argument('--video_path', type=str, default=default_video, help="Path to input video file or folder (e.g., translation_target/)")
    parser.add_argument('--ckpt_path', type=str, default=default_ckpt, help="Path to SpaMo checkpoint")
    parser.add_argument('--config_path', type=str, default=default_cfg, help="Path to config file")
    return parser.parse_args()


def extract_and_crop_frames(video_path, frames_dir):
    """
    Reads a video, center-crops to the 210:260 aspect ratio,
    resizes frames to 210x260, and saves them to frames_dir.
    Returns a list of PIL Images.
    """
    import glob
    existing_frames = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    pil_frames = []
    
    if len(existing_frames) > 0:
        print(f"Loading {len(existing_frames)} existing frames from: {frames_dir}")
        for img_path in existing_frames:
            pil_frames.append(Image.open(img_path).convert('RGB'))
        return pil_frames

    print(f"Extracting frames from {video_path}")
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
        
    if not frames:
        raise ValueError("No frames found/extracted.")
        
    target_aspect = TARGET_W / TARGET_H
    
    for idx, frame in enumerate(frames):
        h, w, _ = frame.shape
        video_aspect = w / h
        
        if abs(video_aspect - target_aspect) > 0.01:
            if video_aspect > target_aspect:
                # Video is wider than target. Crop width.
                new_w = int(h * target_aspect)
                start_x = (w - new_w) // 2
                cropped = frame[:, start_x:start_x+new_w]
            else:
                # Video is taller than target. Crop height.
                new_h = int(w / target_aspect)
                start_y = (h - new_h) // 2
                cropped = frame[start_y:start_y+new_h, :]
        else:
            cropped = frame
            
        # Resize to exactly 210x260
        resized = cv2.resize(cropped, (TARGET_W, TARGET_H))
        pil_frame = Image.fromarray(resized)
        pil_frames.append(pil_frame)
        
        # Save frame to disk
        pil_frame.save(os.path.join(frames_dir, f"{idx:04d}.png"))
        
    return pil_frames


@torch.no_grad()
def extract_spatial_features(frames):
    """
    Extracts CLIP ViT-large spatial features using s2wrapping.
    Expected output shape: (num_frames, 1024)
    """
    print("Extracting Spatial Features (CLIP ViT-Large)...")
    model_name = 'openai/clip-vit-large-patch14'
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = CLIPVisionModel.from_pretrained(model_name, output_hidden_states=True).to(DEVICE).eval()
    
    def forward_features(inputs):
        return model(inputs).hidden_states[-1]
        
    # Process in batches to avoid OOM
    batch_size = 32
    all_feats = []
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        inputs = processor(batch, return_tensors="pt").to(DEVICE).pixel_values
        # s2wrapping with scales=[1, 2]
        outputs = multiscale_forward(forward_features, inputs, scales=[1, 2], num_prefix_token=1)
        # Get CLS token
        cls_feats = outputs[:, 0].cpu().numpy()
        all_feats.append(cls_feats)
        
    del model
    torch.cuda.empty_cache()
    
    return np.concatenate(all_feats, axis=0)


@torch.no_grad()
def extract_motion_features(frames):
    """
    Extracts VideoMAE-large motion features with temporal window of 16 and overlap of 8.
    Expected output shape: (num_windows, 1024)
    """
    print("Extracting Motion Features (VideoMAE-Large)...")
    model_name = 'MCG-NJU/videomae-large'
    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    model = VideoMAEModel.from_pretrained(model_name).to(DEVICE).eval()
    
    # Needs at least 16 frames
    if len(frames) < 16:
        frames.extend([frames[-1]] * (16 - len(frames)))
        
    # Split into sliding windows (size 16, overlap 8)
    windows = sliding_window_for_list(frames, window_size=16, overlap_size=8)
    
    batch_size = 16
    all_feats = []
    
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i+batch_size]
        # batch is a list of lists of images
        inputs = processor(images=batch, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs, output_hidden_states=True).hidden_states[-1]
        # Get CLS token
        cls_feats = outputs[:, 0].cpu().numpy()
        all_feats.append(cls_feats)
        
    del model
    torch.cuda.empty_cache()
    
    return np.concatenate(all_feats, axis=0)


def load_spamo_model(ckpt_path, config_path):
    print("Loading SpaMo model...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Build model using config params
    model = FlanT5SLT(**config['model']['params'])
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
        
    # Remove "model." prefix from state dict if it's there (PyTorch Lightning artifact)
    clean_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if not k.startswith('t5_model')}
    
    # Handle PEFT/LoRA weights
    model.load_state_dict(clean_state_dict, strict=False)
    
    peft_weights = {k.replace('t5_model.', ''): v for k, v in state_dict.items() if 'lora' in k}
    if peft_weights:
        # Use direct load_state_dict instead of set_peft_model_state_dict
        # (the PEFT convenience function silently drops weights)
        model.t5_model.load_state_dict(peft_weights, strict=False)
        print(f"LoRA adapter applied to T5 model. ({len(peft_weights)} weight tensors loaded)")
        
    model = model.to(DEVICE).eval()
    return model


@torch.no_grad()
def translate(model, spatial_feats, motion_feats):
    """
    Run translation using the model's NATIVE evaluation path.
    Mirrors shared_step: prepare_visual_inputs -> fusion_proj -> prepare_inputs -> generate
    """
    print("Running Translation...")
    
    # Convert numpy to tensors (2D, NOT pre-batched)
    # The model's get_inputs returns pixel_values as a LIST of 2D tensors
    spatial_tensor = torch.tensor(spatial_feats, dtype=torch.float32).to(DEVICE)
    motion_tensor = torch.tensor(motion_feats, dtype=torch.float32).to(DEVICE)
    
    # Build batch exactly as get_inputs() returns it:
    # pixel_values and glor_values are LISTS of 2D tensors (one per sample)
    # num_frames and glor_lengths are LISTS of ints
    inputs = {
        'pixel_values': [spatial_tensor],          # List of 2D tensors
        'glor_values': [motion_tensor],            # List of 2D tensors
        'num_frames': [spatial_tensor.shape[0]],   # List of ints
        'glor_lengths': [motion_tensor.shape[0]],  # List of ints
        'ids': ['inference_video'],
        'text': ['placeholder.'],                  # Needed by prepare_inputs
        'lang': ['German'],                        # Model was trained on German
        'ex_lang_trans': [''],                     # Empty in-context
        'gloss': [''],
    }
    
    # Temporarily disable in-context learning for inference
    original_use_in_context = model.use_in_context
    model.use_in_context = False
    
    # Step 1: prepare_visual_inputs
    #   - Projects spatial (2048->768) and motion (1024->768) features
    #   - Concatenates them
    #   - Runs temporal_encoder (Conv1d + MaxPool, reduces sequence length)
    #   - Returns 768-dim features
    visual_outputs, visual_masks = model.prepare_visual_inputs(inputs)
    print(f"  Visual outputs after prepare_visual_inputs: shape={visual_outputs.shape}")
    
    # Step 2: fusion_proj (as done in shared_step line 446)
    #   - Maps 768-dim -> 2048-dim (T5 hidden size)
    visual_outputs = model.fusion_proj(visual_outputs)
    print(f"  Visual outputs after fusion_proj: shape={visual_outputs.shape}")
    
    # Step 3: prepare_inputs (as done in shared_step for eval)
    #   - Tokenizes prompt
    #   - Concatenates visual embeddings with prompt embeddings per sample
    #   - Creates proper attention masks
    input_embeds, input_masks, _, _ = model.prepare_inputs(
        visual_outputs, visual_masks, inputs, 'test', 0
    )
    print(f"  Combined input_embeds: shape={input_embeds.shape}")
    
    # Step 4: Generate (exactly as done in shared_step eval)
    generated = model.t5_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=input_masks,
        num_beams=5,
        max_length=model.max_txt_len,
        do_sample=False,
        length_penalty=1.0,
    )
    
    # Step 5: Decode
    translation = model.t5_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    
    # Restore model state
    model.use_in_context = original_use_in_context
    
    return translation


def main():
    args = get_args()
    
    import glob
    video_path = os.path.abspath(args.video_path)
    if os.path.isdir(video_path):
        video_files = glob.glob(os.path.join(video_path, "*.mp4")) + \
                      glob.glob(os.path.join(video_path, "*.avi")) + \
                      glob.glob(os.path.join(video_path, "*.mkv"))
        if not video_files:
            raise ValueError(f"No video files found in directory: {video_path}")
        video_path = video_files[0]
        print(f"Found video inside directory: {video_path}")
        
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    base_target_dir = os.path.dirname(video_path)
    
    video_dir = os.path.join(base_target_dir, video_name)
    frames_dir = os.path.join(video_dir, "frames")
    features_dir = os.path.join(video_dir, "features")
    
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    
    spatial_path = os.path.join(features_dir, "spatial.npy")
    motion_path = os.path.join(features_dir, "motion.npy")

    # 1. Extract and Crop Frames
    print(f"Processing video: {video_path}")
    frames = extract_and_crop_frames(video_path, frames_dir)
    print(f"Frames ready: {len(frames)} frames at {TARGET_W}x{TARGET_H}.")
    
    # 2. Extract Spatial Features
    if os.path.exists(spatial_path):
        print(f"Loading cached Spatial Features from: {spatial_path}")
        spatial_feats = np.load(spatial_path)
    else:
        spatial_feats = extract_spatial_features(frames)
        np.save(spatial_path, spatial_feats)
        print(f"Saved Spatial Features to: {spatial_path}")
    print(f"Spatial features shape: {spatial_feats.shape} (Expected: N x 1024)")
    
    # 3. Extract Motion Features
    if os.path.exists(motion_path):
        print(f"Loading cached Motion Features from: {motion_path}")
        motion_feats = np.load(motion_path)
    else:
        motion_feats = extract_motion_features(frames)
        np.save(motion_path, motion_feats)
        print(f"Saved Motion Features to: {motion_path}")
    print(f"Motion features shape: {motion_feats.shape} (Expected: M x 1024)")
    
    # 4. Load SpaMo
    model = load_spamo_model(args.ckpt_path, args.config_path)
    
    # 5. Translate
    translation = translate(model, spatial_feats, motion_feats)
    
    # 6. Translate German -> English using the same T5 model
    with torch.no_grad():
        en_input = model.t5_tokenizer(
            f"Translate German to English: {translation}",
            return_tensors="pt", max_length=256, truncation=True
        ).to(DEVICE)
        en_output = model.t5_model.generate(**en_input, max_length=256, num_beams=5)
        english = model.t5_tokenizer.decode(en_output[0], skip_special_tokens=True)
    
    print("\n" + "="*50)
    print("FINAL TRANSLATION (German):")
    print(translation)
    print("\nENGLISH:")
    print(english)
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
