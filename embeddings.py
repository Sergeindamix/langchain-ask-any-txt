import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

def extract_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    for script in soup(['script']):
        script.extract()
    return soup.get_text().lower()

def run_question_answering(url):
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
