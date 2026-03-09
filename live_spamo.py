import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer, BitsAndBytesConfig,
    CLIPVisionModel, VideoMAEModel, CLIPProcessor, VideoMAEImageProcessor
)
from peft import LoraConfig, get_peft_model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIG ---
LLM_PATH = "./models/flan-t5-xl"
SPA_CKPT = "./weights/spamo.ckpt"
DEVICE = "cuda"

# === FLAGS ===
ENABLE_GRADCAM = True  # Set to False to disable Grad-CAM visualization
BACKGROUND_COLOR = (0, 0, 0)  # Black background like PHOENIX-14T

# --- MediaPipe Image Segmentation Setup (New API) ---
# Download the model if not present
import urllib.request
import os

MODEL_PATH = "selfie_segmenter.tflite"
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading MediaPipe segmentation model...")
    url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("✅ Model downloaded!")

# Create segmenter
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ImageSegmenterOptions(
    base_options=base_options,
    output_category_mask=True,
    running_mode=vision.RunningMode.VIDEO
)
segmenter = vision.ImageSegmenter.create_from_options(options)

frame_timestamp_ms = 0

def remove_background(frame_bgr):
    """Remove background using MediaPipe and replace with solid color"""
    global frame_timestamp_ms
    
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # Process with timestamp for VIDEO mode
    frame_timestamp_ms += 33  # ~30fps
    result = segmenter.segment_for_video(mp_image, frame_timestamp_ms)
    
    # Get category mask
    if result.category_mask is None:
        return frame_bgr
    
    mask = result.category_mask.numpy_view()
    
    # Selfie segmenter: 0 = background, 255 = person (or 1 = person depending on model)
    # Normalize mask to 0-1 range
    if mask.max() > 1:
        binary_mask = (mask / 255.0).astype(np.float32)
    else:
        binary_mask = mask.astype(np.float32)
    
    # Mask is inverted - flip it so person=1, background=0
    binary_mask = 1.0 - binary_mask
    
    # Smooth edges with gaussian blur
    binary_mask = cv2.GaussianBlur(binary_mask, (7, 7), 0)
    
    # Threshold to clean up edges
    binary_mask = np.where(binary_mask > 0.5, 1.0, 0.0).astype(np.float32)
    binary_mask = cv2.GaussianBlur(binary_mask, (5, 5), 0)  # Smooth again after threshold
    
    mask_3ch = np.stack([binary_mask] * 3, axis=-1)
    
    # Create background
    bg = np.full_like(frame_bgr, BACKGROUND_COLOR, dtype=np.uint8)
    
    # Composite: foreground * mask + background * (1 - mask)
    # Person (mask=1) shows original, Background (mask=0) shows black
    output = (frame_bgr * mask_3ch + bg * (1 - mask_3ch)).astype(np.uint8)
    
    return output

# --- WRAPPERS TO FIX ATTRIBUTEERROR ---
class CLIPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).pooler_output

class VideoMAEWrapper(nn.Module):
    """Wrapper that reshapes 5D video input to 4D for Grad-CAM compatibility"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._input_shape = None
    
    def forward(self, pixel_values):
        self._input_shape = pixel_values.shape
        return self.model(pixel_values=pixel_values).last_hidden_state.mean(dim=1)

# --- ARCHITECTURE (Matched to your .ckpt) ---
class SignAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatio_proj = nn.Linear(2048, 768)
        self.spatiotemp_proj = nn.Linear(1024, 768)
        self.fusion_proj = nn.Sequential(
            nn.Linear(768, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048)
        )
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(768, 768, kernel_size=3, padding=1),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Conv1d(768, 768, kernel_size=3, padding=1),
            nn.BatchNorm1d(768)
        )

print(f"🚀 Launching SpaMo Live...")
print(f"   Grad-CAM: {'ENABLED' if ENABLE_GRADCAM else 'DISABLED'}")
print(f"   Background: {BACKGROUND_COLOR}")

# 1. Load Brain (4-bit BF16)
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
base_model = T5ForConditionalGeneration.from_pretrained(LLM_PATH, quantization_config=bnb_config, local_files_only=True, device_map="auto")
tokenizer = T5Tokenizer.from_pretrained(LLM_PATH, local_files_only=True)
model = get_peft_model(base_model, LoraConfig(r=16, lora_alpha=32, target_modules=["q", "v"], task_type="SEQ_2_SEQ_LM"))

# 2. Load Adapter
adapter = SignAdapter().to(DEVICE).to(torch.bfloat16)
checkpoint = torch.load(SPA_CKPT, map_location="cpu")
model.load_state_dict({k.replace("t5_model.base_model.model.", ""): v for k, v in checkpoint['state_dict'].items() if "t5_model" in k}, strict=False)

# 3. Load Encoders & WRAP them
spatial_enc_raw = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()
motion_enc_raw = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").to(DEVICE).eval()

spatial_enc = CLIPWrapper(spatial_enc_raw)
motion_enc = VideoMAEWrapper(motion_enc_raw)

spatial_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
motion_proc = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

# 4. Grad-CAM Setup (only if enabled)
spatial_cam = None
motion_target = None

if ENABLE_GRADCAM:
    def clip_reshape_transform(tensor):
        result = tensor[:, 1:, :].reshape(tensor.size(0), 16, 16, tensor.size(2))
        return result.permute(0, 3, 1, 2)

    def mae_reshape_transform(tensor):
        batch_size = tensor.size(0)
        num_tokens = tensor.size(1)
        hidden_dim = tensor.size(2)
        num_temporal = num_tokens // (14 * 14)
        result = tensor.reshape(batch_size, num_temporal, 14, 14, hidden_dim)
        result = result.mean(dim=1)
        return result.permute(0, 3, 1, 2)

    spatial_target = spatial_enc_raw.vision_model.encoder.layers[-1].layer_norm1
    motion_target = motion_enc_raw.encoder.layer[-1].layernorm_before

    print(f"Spatial target layer: {spatial_target}")
    print(f"Motion target layer: {motion_target}")

    spatial_cam = GradCAM(
        model=spatial_enc, 
        target_layers=[spatial_target], 
        reshape_transform=clip_reshape_transform
    )

    def compute_videomae_gradcam(model, input_tensor, target_layer):
        """Manually compute Grad-CAM for VideoMAE to avoid the 5D input issue"""
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output.detach())
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())
        
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        try:
            model.zero_grad()
            output = model(input_tensor)
            target = output.mean()
            target.backward()
            
            act = activations[0]
            grad = gradients[0]
            
            weights = grad.mean(dim=1, keepdim=True)
            cam = (weights * act).sum(dim=-1)
            
            batch_size = cam.size(0)
            num_patches = cam.size(1)
            num_temporal = num_patches // (14 * 14)
            
            cam = cam.reshape(batch_size, num_temporal, 14, 14)
            cam = cam.mean(dim=1)
            
            cam = torch.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            
            return cam.cpu().numpy()
        
        finally:
            forward_handle.remove()
            backward_handle.remove()

# --- WEBCAM LOOP ---
cap = cv2.VideoCapture(0)
frame_buffer = []
last_translation = ""
spatial_heatmap = None
motion_heatmap = None

print("📷 Starting webcam... Press 'q' to quit, 'g' to toggle Grad-CAM.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Remove background
    processed_frame = remove_background(frame)
    
    # Convert to PIL for processors
    rgb_processed = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    frame_buffer.append(Image.fromarray(rgb_processed))

    # Display Windows - all show the processed (background removed) feed
    display_frame = processed_frame.copy()
    cv2.putText(display_frame, f"DGS: {last_translation}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Buffer: {len(frame_buffer)}/64", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(display_frame, f"Grad-CAM: {'ON' if ENABLE_GRADCAM else 'OFF'} (press 'g')", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.imshow('1. Processed Feed', display_frame)
    
    if ENABLE_GRADCAM:
        if spatial_heatmap is not None: 
            cv2.imshow('2. Spatial Attention (CLIP)', spatial_heatmap)
        if motion_heatmap is not None: 
            cv2.imshow('3. Motion Attention (MAE)', motion_heatmap)
    else:
        try:
            cv2.destroyWindow('2. Spatial Attention (CLIP)')
            cv2.destroyWindow('3. Motion Attention (MAE)')
        except:
            pass

    if len(frame_buffer) >= 64:
        print("🔄 Processing 64 frames...")
        
        try:
            # Get middle frame for display
            middle_frame = frame_buffer[32]
            middle_np = np.array(middle_frame)
            middle_bgr = cv2.cvtColor(middle_np, cv2.COLOR_RGB2BGR)
            middle_bgr_resized = cv2.resize(middle_bgr, (frame.shape[1], frame.shape[0]))
            
            # Prepare inputs
            s_in = spatial_proc(images=middle_frame, return_tensors="pt")
            s_in = {k: v.to(DEVICE) for k, v in s_in.items()}
            
            subsampled = [frame_buffer[i] for i in range(0, 64, 4)]
            m_in = motion_proc(subsampled, return_tensors="pt")
            m_in = {k: v.to(DEVICE) for k, v in m_in.items()}

            # A. Calculate Grad-CAMs (if enabled)
            if ENABLE_GRADCAM and spatial_cam is not None:
                # 1. Spatial CAM
                s_mask = spatial_cam(input_tensor=s_in['pixel_values'], targets=None)[0]
                s_mask_resized = cv2.resize(s_mask, (frame.shape[1], frame.shape[0]))
                spatial_heatmap = show_cam_on_image(np.float32(middle_bgr_resized)/255, s_mask_resized, use_rgb=False)

                # 2. Motion CAM
                m_mask = compute_videomae_gradcam(
                    motion_enc, 
                    m_in['pixel_values'], 
                    motion_target
                )[0]
                m_mask_resized = cv2.resize(m_mask, (frame.shape[1], frame.shape[0]))
                motion_heatmap = show_cam_on_image(np.float32(middle_bgr_resized)/255, m_mask_resized, use_rgb=False)

            # B. Translation
            with torch.no_grad():
                s_feats = spatial_enc_raw(**s_in).last_hidden_state
                m_feats = motion_enc_raw(**m_in).last_hidden_state
                
                # 1. Temporal Processing
                m_input_adapter = m_feats.transpose(1, 2).to(torch.bfloat16)
                m_encoded = adapter.temporal_encoder(m_input_adapter).mean(dim=2)
                
                # 2. Spatial Processing
                s_pooled = s_feats.mean(dim=1).to(torch.bfloat16)
                
                # 3. SIGNAL BOOST: Normalize the 'volume' of both encoders
                # This prevents one encoder from 'drowning out' the other
                s_pooled = torch.nn.functional.normalize(s_pooled, dim=-1)
                m_encoded = torch.nn.functional.normalize(m_encoded, dim=-1)
                
                # 4. WEIGHTED FUSION: Give motion (MAE) the lead role (70% weight)
                # Your Grad-CAM showed MAE was distracted; this forces its importance
                fused_mid = (0.3 * adapter.spatiotemp_proj(s_pooled)) + (0.7 * m_encoded)
                fused_final = adapter.fusion_proj(fused_mid).unsqueeze(1)

                # 5. TWO-STAGE TRANSLATION: DGS -> German -> English
                # Stage 1: Get German gloss from sign language
                prompt_dgs = "übersetzen: " 
                text_ids = tokenizer(prompt_dgs, return_tensors="pt").input_ids.to(DEVICE)
                text_embeds = model.get_input_embeddings()(text_ids)
                full_embeds = torch.cat([fused_final, text_embeds], dim=1)

                outputs_german = model.generate(
                    inputs_embeds=full_embeds, 
                    max_length=15,
                    num_beams=8,
                    repetition_penalty=3.5,
                    length_penalty=1.0,
                    early_stopping=True
                )
                german_text = tokenizer.decode(outputs_german[0], skip_special_tokens=True)
                german_text = german_text.replace("übersetzen:", "").strip()
                
                # Stage 2: Translate German to English using T5's built-in capability
                if german_text:
                    translate_prompt = f"translate German to English: {german_text}"
                    translate_ids = tokenizer(translate_prompt, return_tensors="pt").input_ids.to(DEVICE)
                    
                    outputs_english = model.generate(
                        input_ids=translate_ids,
                        max_length=30,
                        num_beams=4,
                        early_stopping=True
                    )
                    english_text = tokenizer.decode(outputs_english[0], skip_special_tokens=True)
                    
                    if english_text != last_translation:
                        print(f"📝 German: {german_text}")
                        print(f"📝 English: {english_text}")
                        last_translation = english_text

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

        # Keep last 32 frames for overlap
        frame_buffer = frame_buffer[32:]
    
    # Key handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('g'):
        ENABLE_GRADCAM = not ENABLE_GRADCAM
        print(f"🔄 Grad-CAM: {'ENABLED' if ENABLE_GRADCAM else 'DISABLED'}")
        if not ENABLE_GRADCAM:
            spatial_heatmap = None
            motion_heatmap = None

cap.release()
segmenter.close()
cv2.destroyAllWindows()
print("👋 Done!")