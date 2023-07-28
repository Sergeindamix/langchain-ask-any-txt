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

    # create replicate model with the specified image dimensions
    model = Replicate(
        model="stability-ai/sdxl:2b017d9b67edd2ee1401238df49d75da53c523f36e363881e057f5dc3ed3c5b2",
        input={"image_dimensions": image_dimensions}
    )

    # get user input
    input_text = st.text_input("Enter a description of the image")

    if input_text:
        # generate image using replicate model
        image_url = model(input_text)

        # download and display image
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        st.image(img, caption=input_text)

if __name__ == "__main__":
    main()
