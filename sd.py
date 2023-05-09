#!pip install replicate
# get a token: https://replicate.com/account 



REPLICATE_API_TOKEN = "r8_brmwdLDHOD3znpSbSdz8OeUQfTLXujD1ttixD"
import os

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN #"r8_brmwdLDHOD3znpSbSdz8OeUQfTLXujD1ttixD"
from langchain.llms import Replicate
from langchain import PromptTemplate, LLMChain
import streamlit as st
#from replicate import Replicate
from PIL import Image
import requests
from io import BytesIO



def sd(input_text):
    st.title("Text to Image Conversion")

    # create replicate model
    model = Replicate(
        model="stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
        input={"image_dimensions": "768x768"}
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
