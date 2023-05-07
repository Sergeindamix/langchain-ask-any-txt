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

      
eleven_api_key = "cb5bafe60ce95aa3cd258c16bc2b1a4d"


voice_list = voices()
voice_labels = [voice.category + " voice: " + voice.name for voice in voice_list]

voice_id = st.selectbox("Selecciona una voz:", voice_labels)

def generate_audio(text, speed):
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

    headers = {
        "Accept": "audio/mp3",
        "xi-api-key": eleven_api_key
    }

    data = {
        "text": text,
        "voice_settings": {
            "speed": speed
        }
    }

    response = requests.post(tts_url, json=data, headers=headers, stream=True)

    if response.status_code == 200:
        # convert the streamed mp3 data to a playable audio format
        audio_data = BytesIO(response.content)
        audio_segment = AudioSegment.from_file(audio_data, format="mp3")
        return audio_segment
    else:
        st.warning(f"No se pudo generar el audio. El servidor devolvió el código de estado {response.status_code}.")
        return None




def play_audio_file(file_path):
    audio = AudioSegment.from_file(file_path)
    play_audio(audio)

def main():
    load_dotenv()
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



          
      st.write(response)
      
def ytsub():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type=["pdf", "docx", "txt"])
    
    # extract the text
    if pdf is not None:
        if pdf.type == 'application/pdf':
            with open('uploaded_file.pdf', 'wb') as f:
                f.write(pdf.read())
            if os.path.exists('uploaded_file.pdf'):
                text = textract.process('uploaded_file.pdf').decode('utf-8')
                os.remove('uploaded_file.pdf')
            else:
                # handle file not found error
                ...
        else:
            # handle non-PDF files
            with open('uploaded_file', 'wb') as f:
                f.write(pdf.read())
            if os.path.exists('uploaded_file'):
                text = textract.process('uploaded_file').decode('utf-8')
                os.remove('uploaded_file')
            else:
                # handle file not found error
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


if __name__ == '__main__':
    main()
