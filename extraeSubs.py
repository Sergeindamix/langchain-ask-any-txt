import streamlit as st
import re
import os
import textract
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

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


# Save the extracted text to a file
with open("ronaldo.txt", "w") as f:
    f.write(text)

# Load the text using TextLoader
loader = TextLoader("ronaldo.txt")
document = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"])
docs = text_splitter.split_documents(document)

# Create the FAISS vector store
embedding = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(docs, embedding)

# Display the docs
st.write(vector_store)
