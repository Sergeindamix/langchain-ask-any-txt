import streamlit as st
import requests
from IPython.display import Audio
import soundfile as sf
from transformers.tools import HfAgent
from gtts import gTTS
# InstructorEmbedding 
from langchain.embeddings import HuggingFaceInstructEmbeddings
import faiss
import pickle
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModel
import torch

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")

# Definir el modelo y el tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def text_downloader(url):
    response = requests.get(url)
    if response.status_code == 200:
        # split into chunks    
        chunks = split_text(response.text)
        st.write(chunks)
        
        
          
        return response.text
    else:
        return ""


# Define una función para dividir el texto en trozos de 2048 tokens
def split_text(text):
  # Divide el texto en trozos de 512 tokens
  tokens = text.split()
  chunks = []
  for i in range(0, len(tokens), 512):
      chunk = ' '.join(tokens[i:i+512])
      chunks.append(chunk)
  return chunks


def text_reader(text):    
    #audio = agent.run(text) #"Please read out loud the contents of the https://gamma.app/docs/sgx5tfyh0wttucu")
    # Convierte el texto en un archivo de audio
    tts = gTTS(text=text, lang="es")
    audio_file = "audio.mp3"
    tts.save(audio_file)
    # Reproduce el archivo de audio
    return Audio(audio_file, autoplay=True)

def promptx():
    # Define el título de la aplicación
    st.title("Convertidor de texto a voz")
    # Pide al usuario que ingrese la URL del artículo
    url = st.text_input("Ingresa la URL del artículo:")
    # Descarga el texto del artículo
    text = text_downloader(url)
    # Muestra el texto del artículo
    st.write("Texto del artículo:")
    st.write(text)
    

    
    if st.button("Download Text"):
        text = text_downloader(url)
        st.write(f"The text is {text}")

    if st.button("Generate Audio"):
      # Convierte el texto en un archivo de audio y lo reproduce
      st.write("Audio del artículo:")
      audio = text_reader(text)
      st.write(audio)
        
        
if __name__ == "__main__":
    promptx()
