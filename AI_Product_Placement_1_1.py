import torch
import numpy as np
import cv2
from diffusers import StableDiffusionXLImg2ImgPipeline, ControlNetModel, StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from rembg import remove
from PIL import Image
import os
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Room decor placement using AI")
    parser.add_argument("--input_dir", required=True, help="Directory containing decor images")
    parser.add_argument("--output_dir", required=True, help="Directory to save output images")
    parser.add_argument("--device", default=None, help="Device to use: 'cuda', 'mps', or 'cpu'")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # SETUP
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device}")

    # Load Stable Diffusion 2.1 for Room Generation
    sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)

    # Load ControlNet for Depth Mapping
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=dtype).to(device)
    controlnet_pipe.scheduler = DPMSolverMultistepScheduler.from_config(controlnet_pipe.scheduler.config)

    # Load Depth Estimation Model
    depth_estimator = pipeline("depth-estimation")

    # Process each decor image
    batch_process_decor(args.input_dir, args.output_dir, sd_pipe, controlnet_pipe, depth_estimator)

def generate_lifestyle_room(sd_pipe, prompt, output_path):
    """Generate a lifestyle room and save it."""
    image = sd_pipe(prompt=prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image.save(output_path)
    return image

def remove_background(input_path, output_path):
    """Remove background from decor images."""
    image = Image.open(input_path)
    image = remove(image)
    image.save(output_path, "PNG")

def estimate_depth(image_path, output_path, depth_estimator):
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

def generate_depth_guided_room(controlnet_pipe, room_path, depth_path, prompt, output_path):
    """Generate room using depth-guided control net."""
    room_image = Image.open(room_path).convert("RGB")
    depth_image = Image.open(depth_path).convert("RGB")
    output = controlnet_pipe(image=room_image, prompt=prompt, control_image=depth_image, num_inference_steps=50, guidance_scale=7.5).images[0]
    output.save(output_path)
    return output

def auto_place_decor(room_path, decor_path, mask_path, output_path):
    """Automatically place decor item into the generated room."""
    room = cv2.imread(room_path)
    decor = cv2.imread(decor_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])  # AI-selected position

        # Get original decor size
        dh, dw = decor.shape[:2]

        # Keep aspect ratio when resizing
        scale = min(w / dw, h / dh)  # Fit inside the mask box
        scale = max(scale, 0.7)  # Prevent over-shrinking (minimum 50% size)
        scale = min(scale, 1.0)  # Prevent under-shrinking (maximum 100% size)
        new_w, new_h = int(dw * scale), int(dh * scale)
        decor_resized = cv2.resize(decor, (new_w, new_h))

        # Ensure decor has an alpha channel
        if decor_resized.shape[-1] == 3:
            alpha_channel = np.ones((new_h, new_w), dtype=np.uint8) * 255  # Add full opacity
            decor_resized = np.dstack([decor_resized, alpha_channel])

        # Center decor in the mask area
        x1 = x + (w - new_w) // 2
        y1 = y + (h - new_h) // 2
        x2, y2 = x1 + new_w, y1 + new_h

        # Ensure decor fits within room bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(room.shape[1], x2)  # room width
        y2 = min(room.shape[0], y2)  # room height

        # Resize decor again to fit inside the new bounding box
        new_w, new_h = x2 - x1, y2 - y1
        if new_w > 0 and new_h > 0:
            decor_resized = cv2.resize(decor_resized, (new_w, new_h))

            # Extract alpha mask for blending
            alpha_s = decor_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            # Blend images
            for c in range(3):  # Only RGB channels
                room[y1:y2, x1:x2, c] = (alpha_s * decor_resized[:, :, c] + alpha_l * room[y1:y2, x1:x2, c])

        cv2.imwrite(output_path, room)

def batch_process_decor(input_dir, output_dir, sd_pipe, controlnet_pipe, depth_estimator):
    """Run the full process individually for each decor image."""
    for decor_file in os.listdir(input_dir):
        if decor_file.endswith((".png", ".jpg", ".jpeg")):
            decor_path = os.path.join(input_dir, decor_file)
            
            # Unique filenames for each decor item
            room_output = os.path.join(output_dir, f"room_{decor_file}.png")
            depth_output = os.path.join(output_dir, f"depth_{decor_file}.png")
            mask_output = os.path.join(output_dir, f"mask_{decor_file}.png")
            depth_guided_output = os.path.join(output_dir, f"depth_guided_{decor_file}.png")
            processed_decor_path = os.path.join(output_dir, f"processed_{decor_file}")
            final_output = os.path.join(output_dir, f"final_{decor_file}")

            # Generate room and process decor separately for each item
            generate_lifestyle_room(sd_pipe, prompt="A cozy modern living room with a sofa and plants", output_path=room_output)
            estimate_depth(room_output, depth_output, depth_estimator)
            generate_decor_mask(depth_output, mask_output)
            generate_depth_guided_room(controlnet_pipe, room_output, depth_output, prompt="A cozy modern living room with a sofa and plants", output_path=depth_guided_output)
            remove_background(decor_path, processed_decor_path)
            auto_place_decor(depth_guided_output, processed_decor_path, mask_output, final_output)

if __name__ == "__main__":
    main()
