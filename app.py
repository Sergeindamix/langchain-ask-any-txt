from youtube_transcript_api import YouTubeTranscriptApi
import re
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain import memory
from langchain.chains.conversation.base import ConversationChain
from langchain.chat_models import ChatOpenAI 
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
import os
import textract
import requests
import io
from pydub import AudioSegment
from IPython.display import Audio, clear_output
import tempfile
import IPython
import IPython.display as ipd
from elevenlabs import generate, play, set_api_key, voices, Models
from pydub.playback import play as play_audio
from io import BytesIO
import base64
import docx2txt
from sd import sd
from LAVIS import imgx
from agentes import promptx
from embds import embds

st.set_page_config(page_title="🦜🔗 Ask YouTube or Docs💬")
st.header("🦜🔗 Ask YouTube or Docs💬")

# Create a toggle widget to generate image
show_text = st.checkbox("Generar imagen?", value=False)

# Display some text if the toggle is on
if show_text:
    # get user input
    input_text = ""

    sd(input_text)
    promptx()
    img_path = "1.jpg"
    imgx(img_path)

from transformers.tools import HfAgent
agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
text = "The main topic of this text is the benefits of exercise for overall health and well-being. Studies have shown that regular physical activity can help reduce the risk of chronic diseases such as heart disease, diabetes, and cancer, as well as improve mental health and cognitive function."
    
prompts = ['"Answer the question in the variable `question` about the image stored in the variable `image`. The question is in Spanish."', 
'"Identify the oldest person in the `document` and create an image showcasing the result."', 
'"Generate an image using the text given in the variable `caption`."', 
'f"Summarize the text given in {text} and read it out loud."', 
'"Answer the question in the variable `question` about the text in the variable `text`. Use the answer to generate an image."', 
'"Caption the following `image`."', 
'"<<prompt>>"']
prompt = st.selectbox("Selecciona un prompt:", prompts)

def download_file(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode('utf-8')
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_path}">Descargar archivo</a>'
    return href

def downloadDoc(user_question, response):
  import docx
      
  # Crear nuevo documento de Word
  document = docx.Document()

  # Agregar título
  document.add_heading("Respuesta", level=1)

  # Agregar subtítulo
  document.add_heading("Pregunta del usuario:", level=2)
  document.add_paragraph(user_question)

  # Agregar subtítulo
  document.add_heading("Respuesta:", level=2)
  document.add_paragraph(response)

  # Guardar documento en un archivo
  document.save("respuesta_0.docx")

  

  counter = 0

  while os.path.exists(f"respuesta_{counter}.docx"):
      counter += 1

  document.save(f"respuesta_{counter}.docx")

  st.write(download_file(f"respuesta_{counter}.docx"), unsafe_allow_html=True)
      






def main():
    load_dotenv()
    st.header("🦜🔗 YouTube GPT💬")
    url = st.text_input("Ingresa link de YouTube, ejemplo: https://www.youtube.com/watch?v=KczJNtexinY")
    
    # extract video ID using regular expression
    match = re.search(r"v=(\w+)", url)
    # asigna la variable 'video_id' con el id del video de YouTube
    video_id = None

    if "youtu.be/" in url:
        # extract video id from "https://youtu.be/video_id" format
        video_id = url.split("youtu.be/")[-1]
    elif "watch?v=" in url:
        # extract video id from "https://www.youtube.com/watch?v=video_id" format
        video_id = url.split("watch?v=")[-1]

    # asigna la variable 'language_code' con el código de idioma deseado
    idioma = st.selectbox("Selecciona el idioma de los subtitulos, si no existen mostrará error", ["en", "es", "fr"])

    # obtiene el transcript en el idioma deseado
    srt = YouTubeTranscriptApi.get_transcript(video_id, languages=[idioma])
    # eliminar los caracteres de formato
    # extraer solo el texto de los subtítulos
    text = ""
    for subtitle in srt:
        text += subtitle['text'] + " "
    
        
    # eliminar los caracteres de formato
    text = text.replace('\n', ' ').replace('\r', '')
    
    # Lee el archivo de subtítulos
    with open('subtitles.txt', 'w', encoding='utf-8') as file:
      # escribir cada línea de subtítulos en el archivo
      for line in srt:
          text = line['text']
          file.write(text + '\n')

    if os.path.exists('subtitles.txt'):
        text = textract.process('subtitles.txt').decode('utf-8')
        os.remove('subtitles.txt')
    else:
        # handle file not found error
        ...


    # Define una función para dividir el texto en trozos de 2048 tokens
    def split_text(text):
      # Divide el texto en trozos de 512 tokens
      tokens = text.split()
      chunks = []
      for i in range(0, len(tokens), 512):
          chunk = ' '.join(tokens[i:i+512])
          chunks.append(chunk)
      return chunks

        

    # split into chunks    
    chunks = split_text(text)
    st.write(chunks)
    
    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    # show user input
    user_question = st.text_input("Ask a question about YouTube VIDEO:")
    if user_question:    
      docs = knowledge_base.similarity_search(user_question)
      
      llm = OpenAI()
      chain = load_qa_chain(llm, chain_type="stuff")
      with get_openai_callback() as cb:
      
        response = chain.run(input_documents=docs, question=user_question)
        print(cb)

      downloadDoc(user_question, response)

      st.write(response)
      
import transformers
from transformers import pipeline

def is_huggingface_langchain(question, text):
    # initialize the question-answering pipeline
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    # find the answer to the question in the text
    result = qa_pipeline(question=question, context=text)

    # check if the answer contains the words "Hugging Face" and "LangChain"
    answer = result["answer"]
    if "Hugging Face" in answer and "LangChain" in answer:
        st.write(result)
        return True        
    else:
        st.write("error")
        return answer
      
      
def txts():
    load_dotenv()
    #st.set_page_config(page_title="Ask your PDF")
    st.header("[pdf, txt, docx] 💬")
    
    # upload file
    uploaded_file = st.file_uploader("Upload your Document", type=["pdf", "docx", "txt"])
    
    # extract the text
    if uploaded_file is not None:
        if uploaded_file.type == 'application/pdf':
            with open('uploaded_file.pdf', 'wb') as f:
                f.write(uploaded_file.read())
            if os.path.exists('uploaded_file.pdf'):
                text = textract.process('uploaded_file.pdf').decode('utf-8')
                os.remove('uploaded_file.pdf')
            else:
                # handle file not found error
                ...
        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            with open('uploaded_file.docx', 'wb') as f:
                f.write(uploaded_file.read())
            if os.path.exists('uploaded_file.docx'):
                text = docx2txt.process('uploaded_file.docx')
                os.remove('uploaded_file.docx')
        elif uploaded_file.type == 'text/plain':
            # handle non-PDF files            
            with open('uploaded_file.txt', 'wb') as f:
                f.write(uploaded_file.read())
            if os.path.exists('uploaded_file.txt'):
                text = textract.process('uploaded_file.txt').decode('utf-8')
                os.remove('uploaded_file.txt')

                ...
        
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
                
            st.write(response)
            downloadDoc(user_question, response)

txts()


question = "What is the main topic of this text?"
text = "The main topic of this text is the benefits of exercise for overall health and well-being. Studies have shown that regular physical activity can help reduce the risk of chronic diseases such as heart disease, diabetes, and cancer, as well as improve mental health and cognitive function."

response = is_huggingface_langchain(question, text)
st.write(response)

if __name__ == '__main__':
    main()
