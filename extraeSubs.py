import databutton as db
import streamlit as st
import docx2txt
import streamlit as st
import re
import os
import textract
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.vectorstores import SimilaritySearch

# Load environment variables from a .env file if present

st.header("🦜🔗 YouTube GPT💬")
url = st.text_input("Ingresa el enlace de YouTube (ejemplo: https://www.youtube.com/watch?v=KczJNtexinY)")

# Extract video ID using regular expression
match = re.search(r"v=(\w+)", url)
video_id = None

if match:
    video_id = match.group(1)

idioma = st.selectbox("Selecciona el idioma de los subtítulos. Si no existen, mostrará un error.", ["en", "es", "fr"])

try:
    # Get the transcript in the desired language
    srt = YouTubeTranscriptApi.get_transcript(video_id, languages=[idioma])

    # Extract the text from the subtitles
    text = " ".join([subtitle['text'] for subtitle in srt])

    # Remove formatting characters
    text = text.replace('\n', ' ').replace('\r', '')

    # Write subtitles to a temporary file
    with open('subtitles.txt', 'w', encoding='utf-8') as file:
        file.write(text)

    if os.path.exists('subtitles.txt'):
        # Extract text from the temporary file
        text = textract.process('subtitles.txt').decode('utf-8')

        # Remove the temporary file
        os.remove('subtitles.txt')
    else:
        st.error("Error: Subtitles file not found.")
        st.stop()

    # Define a function to split the text into chunks of 512 tokens
    def split_text(text):
        tokens = text.split()
        chunks = []
        for i in range(0, len(tokens), 512):
            chunk = ' '.join(tokens[i:i+512])
            chunks.append(chunk)
        return chunks

    # Split the text into chunks
    chunks = split_text(text)
    st.write(chunks)

    # Create an instance of SimilaritySearch
    similarity_search = SimilaritySearch()

    user_question = st.text_input("Ingresa pregunta")

    # Find similar documents based on the user question and the chunks
    docs = similarity_search.find_similar_documents(user_question, chunks)

    # Process the docs to generate the response and download the document if needed
    response = process_docs(docs)

    # Display the response
    st.write(response)

except Exception as e:
    st.error(f"Error: {str(e)}")
