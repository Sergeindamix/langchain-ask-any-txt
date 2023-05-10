import torch
from PIL import Image

import streamlit as st

def generate_caption(image):
  cation = model.generate({"image": image})


def imgx(image_file):
    st.title("Image Captioning")

    # Allow the user to upload an image
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        # Display the uploaded image
        image = Image.open(image_file)
        st.image(image, caption="Uploaded image", use_column_width=True)

        

if __name__ == "__main__":
    imgx()
