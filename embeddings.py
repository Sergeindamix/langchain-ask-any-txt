import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
import os
import re

def extract_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    for script in soup(['script']):
        script.extract()
    return soup.get_text().lower()


def run_question_answering(url):
    if "youtube.com" in url:
        # Extract video ID using regular expression
        match = re.search(r"v=(\w+)", url)
        if match:
            video_id = match.group(1)
            # asigna la variable 'language_code' con el código de idioma deseado
            idioma = st.selectbox("Selecciona el idioma de los subtitulos, si no existen mostrará error", ["en", "es", "fr"])

            try:
                # Get the transcript in the desired language (e.g., "en" for English)
                srt = YouTubeTranscriptApi.get_transcript(video_id, languages=[idioma])

                # Get the transcript in the desired language (e.g., "en" for English)
                srt = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

                # Extract the text from the subtitles
                text = " ".join([subtitle['text'] for subtitle in srt])

                # Save the extracted text to a file
                with open("subtitles.txt", "w") as f:
                    f.write(text)

                # Load the text using TextLoader
                loader = TextLoader("subtitles.txt")
                document = loader.load()

                # Remove the temporary file
                os.remove("subtitles.txt")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.stop()
        else:
            st.error("Error: Invalid YouTube URL.")
            st.stop()
    else:
        # Extract text from the given URL
        text = extract_text(url)

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
    return vector_store
