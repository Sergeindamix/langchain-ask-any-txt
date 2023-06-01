import requests
import streamlit as st
import os
from urllib.parse import urlparse

def get_filename_from_url(url):
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    return filename

query = st.text_input("Who is the president")
url = st.text_input("https://raw.githubusercontent.com/hwchase17/langchain/master/docs/modules/state_of_the_union.txt")
res = requests.get(url)
filename = get_filename_from_url(url)

with open(filename, "w") as f:
  f.write(res.text)
  #st.write(res.text)

# Document Loader
from langchain.document_loaders import TextLoader
loader = TextLoader(filename)
documents = loader.load()
#st.write(documents)

import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

#st.write(wrap_text_preserve_newlines(str(documents[0])))

# Text Splitter
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

#st.write(len(docs))

# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(docs, embeddings)


docs = db.similarity_search(query)
st.write(wrap_text_preserve_newlines(str(docs[0].page_content)))
