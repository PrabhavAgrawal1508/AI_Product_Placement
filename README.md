# AI_Product_Placement
# Product Lifestyle Image Generator

This tool automatically places indoor decor product images into realistic lifestyle scenes using generative AI. It processes multiple product images sequentially, seamlessly integrating them into diverse backgrounds while preserving product details.

## Features

- **Sequential Processing**: Handle multiple product images sequentially
- **Realistic Placement**: Seamlessly integrate products into diverse lifestyle backgrounds
- **Depth-Guided Integration**: Use depth maps to ensure natural placement with proper perspective
- **Background Removal**: Automatically remove backgrounds from product images
- **Customizable Prompts**: Control the style and content of generated lifestyle rooms

## Requirements

### Hardware
- Any modern computer with at least 8GB RAM
- CUDA-compatible GPU recommended but not required (will run on CPU or Apple Silicon)
- At least 10GB free disk space to incorporate models from Hugging Face

### Software
- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/product-lifestyle-generator.git
   cd product-lifestyle-generator
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the tool with the following command:

```
python main.py --decor_dir /path/to/product/images --output_dir /path/to/output --prompt "A cozy modern living room with a sofa and plants"
```

### Arguments

- `--decor_dir`: Path to the directory containing product images (required)
- `--output_dir`: Path to save the output images (required)
- `--prompt`: Text prompt for generating the lifestyle scene (optional, default: "A cozy modern living room with a sofa and plants")
- `--device`: Device to run the model on, options are "cpu", "cuda", or "mps" (optional, defaults to best available)

## How It Works

1. **Room Generation**: Creates a lifestyle room scene using Stable Diffusion based on the provided prompt
2. **Depth Estimation**: Generates a depth map of the room to understand spatial relationships
3. **Mask Generation**: Creates masks based on depth maps to identify suitable placement areas
4. **Background Removal**: Removes backgrounds from product images using the rembg library
5. **Depth-Guided Integration**: Places products in the room guided by depth maps for realistic perspective

## Technical Approach

This solution leverages several free, open-source AI models and techniques:

- **Stable Diffusion**: For generating the base lifestyle rooms
- **ControlNet with Depth**: For depth-aware image generation and manipulation
- **Depth Estimation**: Using Hugging Face's depth-estimation pipeline
- **Background Removal**: Using the rembg library which implements U2Net

The workflow processes each product image individually through several stages:
1. Generate a lifestyle room scene based on user prompt
2. Estimate the depth map of the generated room
3. Generate a placement mask based on depth discontinuities
4. Create a depth-guided version of the room with proper perspective
5. Remove background from product images to prepare for placement

## Example Results

### Original Product
![Original vase](ExamplePhoto/vase_ex.jpg)

### Final Placement
![Vase placed in room](ExamplePhoto/output.jpeg)


## Limitations and Future Improvements

- Currently optimized for home decor items
- Further work could improve precise positioning of products
- Additional style controls could be implemented
- A user interface could be added for more interactive use

## License

This project uses only free and open-source components with unlimited usage rights.

## Acknowledgments

- StabilityAI for Stable Diffusion models
- Hugging Face for transformers and diffusers libraries
- The rembg project for background removal capabilities

## Contributing
Contributions are welcome! Here to learn more 


## Contact and Connection 
For Contacting me regarding this project or for sharing any kind of Information/Advice/Suggestions
Find me on 
🔗 **LinkedIn:** [Prabhav Agrawal](https://www.linkedin.com/in/prabhav-agrawal-415b83309)  


## Candid Thoughts on This Project
This project marks one of my first deep dives into Generative AI, and the experience has been incredibly eye-opening.

-So far, I have worked exclusively with pre-trained models, as I currently lack access to sufficient trainable data for fine-tuning.
-The computational resources required for running these models are extensive. Thanks to 3 hours and 40 minutes of Google Colab’s T4 GPU access, I was able to bring this project to its current state.
-Through this, I also discovered how poorly optimized these models are for MPS (Metal Shaders) on Apple Silicon devices, leading to constant memory issues, including a semaphore leakage.
-The biggest challenge was developing the object placement logic, which I initially struggled with. I relied on insights from GPT, Claude, and other AI models to refine the approach.
-Despite these efforts, object placement is still far from perfect and remains a key area for future improvements.
-I attempted to use ControlNet Pipeline to directly place decor items into a generated room, but the results were not satisfactory.
-Stable Diffusion XL is too resource-intensive to run efficiently on my MacBook Pro 14" (8-core CPU, 14-core GPU, 16GB RAM).
## Next Steps & Future Plans
Moving forward, I plan to:

- Experiment with Stable Diffusion 2.1 or SDXL using a two-image approach—one for the processed decor item and another for the AI-generated depth-based room—combined with a refined prompt to enhance object placement accuracy.
- Optimize the pipeline further to make this solution viable for commercial marketing applications.
- Continue refining the placement logic to improve realism and adaptability.
## Acknowledgments
I would like to express my gratitude to Viral Nishar and Studio 11 Productions for entrusting me with this project. This experience has been both challenging and fascinating, and I look forward to further developing and refining it in future iterations.