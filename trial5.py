import torch
import numpy as np
import cv2
from diffusers import StableDiffusionXLImg2ImgPipeline, ControlNetModel, StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler , StableDiffusionXLPipeline
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from rembg import remove
from PIL import Image
import os
import argparse

# SETUP
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=None, help="Device to run the model on (cpu, cuda, mps)")
args, _ = parser.parse_known_args()

device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Using device: {device}")

# Load Stable Diffusion XL for Room Generation
sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=dtype,  low_cpu_mem_usage=True).to(device)
sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)

# Load ControlNet for Depth Mapping
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=dtype).to(device)
controlnet_pipe.scheduler = DPMSolverMultistepScheduler.from_config(controlnet_pipe.scheduler.config)

# Load Depth Estimation Model
depth_estimator = pipeline("depth-estimation")

def generate_lifestyle_room(prompt, output_path):
    """Generate a lifestyle room and save it."""
    if not prompt:
        prompt = "A cozy modern living room with a sofa and plants"  # Default prompt
    image = sd_pipe(prompt=prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image.save(output_path)
    return image

def remove_background(input_path, output_path):
    """Remove background from decor images."""
    image = Image.open(input_path)
    image = remove(image)
    image.save(output_path, "PNG")

def estimate_depth(image_path, output_path):
    """Generate depth estimation map."""
    image = Image.open(image_path).convert("RGB")
    depth_map = depth_estimator(image)["depth"]
    depth_map = np.array(depth_map)
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(output_path, depth_map)
    return depth_map
    
def generate_decor_mask(depth_path, output_path):
    """Generate mask based on depth map."""
    depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(depth_img, 30, 100)
    mask = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    cv2.imwrite(output_path, mask)
    return mask

def generate_depth_guided_room(room_path, depth_path, prompt, output_path):
    """Generate room using depth-guided control net."""
    room_image = Image.open(room_path).convert("RGB")
    depth_image = Image.open(depth_path).convert("RGB")
    output = controlnet_pipe(image=room_image, prompt=prompt, control_image=depth_image, num_inference_steps=50, guidance_scale=7.5).images[0]
    output.save(output_path)
    return output

def batch_process_decor(decor_dir, output_dir, prompt):
    """Run the full process individually for each decor image."""
    os.makedirs(output_dir, exist_ok=True)
    
    for decor_file in os.listdir(decor_dir):
        if decor_file.endswith((".png", ".jpg", ".jpeg")):
            decor_path = os.path.join(decor_dir, decor_file)
            
            room_output = os.path.join(output_dir, f"room_{decor_file}.png")
            depth_output = os.path.join(output_dir, f"depth_{decor_file}.png")
            mask_output = os.path.join(output_dir, f"mask_{decor_file}.png")
            depth_guided_output = os.path.join(output_dir, f"depth_guided_{decor_file}.png")
            processed_decor_path = os.path.join(output_dir, f"processed_{decor_file}")
            final_output = os.path.join(output_dir, f"final_{decor_file}")
            
            generate_lifestyle_room(prompt, room_output)
            estimate_depth(room_output, depth_output)
            generate_decor_mask(depth_output, mask_output)
            generate_depth_guided_room(room_output, depth_output, prompt, depth_guided_output)
            remove_background(decor_path, processed_decor_path)

if __name__ == "__main__":
    parser.add_argument("--decor_dir", type=str, required=True, help="Path to decor images directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save output images")
    parser.add_argument("--prompt", type=str, default="A cozy modern living room with a sofa and plants", help="Prompt for room generation")
    args = parser.parse_args()
    
    batch_process_decor(args.decor_dir, args.output_dir, args.prompt)
