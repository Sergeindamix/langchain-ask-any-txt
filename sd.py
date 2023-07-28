# get a token: https://replicate.com/account
REPLICATE_API_TOKEN = "r8_brmwdLDHOD3znpSbSdz8OeUQfTLXujD1ttixD"
import os
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

from langchain.llms import Replicate
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

def sd(input_text):
    st.title("Text to Image Conversion")

    # Text input to get the image dimensions from the user
    image_dimensions = st.text_input("Enter the image dimensions (e.g., 768x768)", "1024x1024")

    # Additional parameters
    negative_prompt = st.text_input("Input Negative Prompt", "anime, cartoon, graphic, text, painting, crayon, graphite, abstract glitch, blurry")
    image_file = st.file_uploader("Input image for img2img or inpaint mode")
    mask_file = st.file_uploader("Input mask for inpaint mode. Black areas will be preserved, white areas will be inpainted.")
    width = st.slider("Width of output image", 1, 2048, 1024)
    height = st.slider("Height of output image", 1, 2048, 1024)
    num_outputs = st.slider("Number of images to output.", 1, 10, 1)
    scheduler = st.selectbox("Scheduler", ["DDIM", "DPMSolverMultistep", "HeunDiscrete", "KarrasDPM", "K_EULER_ANCESTRAL", "K_EULER", "PNDM"], index=0)
    num_inference_steps = st.slider("Number of denoising steps", 1, 100, 50)
    guidance_scale = st.number_input("Scale for classifier-free guidance", value=7.5)
    prompt_strength = st.number_input("Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in the image", value=0.8)
    seed = st.number_input("Random seed. Leave blank to randomize the seed")
    refine = st.selectbox("Which refine style to use", ["no_refiner", "expert_ensemble_refiner", "base_image_refiner"], index=0)
    high_noise_frac = st.number_input("For expert_ensemble_refiner, the fraction of noise to use", value=0.8)
    refine_steps = st.number_input("For base_image_refiner, the number of steps to refine, defaults to num_inference_steps", value=num_inference_steps)

    # create replicate model with the specified parameters
    model = Replicate(
        model="stability-ai/sdxl:2b017d9b67edd2ee1401238df49d75da53c523f36e363881e057f5dc3ed3c5b2",
        input={
            "image_dimensions": image_dimensions,
            "negative_prompt": negative_prompt,
            "image_file": image_file,
            "mask_file": mask_file,
            "width": width,
            "height": height,
            "num_outputs": num_outputs,
            "scheduler": scheduler,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "prompt_strength": prompt_strength,
            "seed": seed,
            "refine": refine,
            "high_noise_frac": high_noise_frac,
            "refine_steps": refine_steps,
        }
    )

    # get user input
    input_text = st.text_input("Enter a description of the image", "breathtaking {prompt} . award-winning, professional, highly detailed")

    if input_text:
        # generate image using replicate model
        image_url = model(input_text)

        # download and display image
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        st.image(img, caption=input_text)

if __name__ == "__main__":
    main()
