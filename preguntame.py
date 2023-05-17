import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub



def answering(vector_store, question):
    # Load the question-answering chain
    llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 0.2, "max_length": 256})
    chain = load_qa_chain(llm, chain_type="stuff")

    # Run the question-answering chain with the given question and input documents
    results = chain.run(input_documents=vector_store.similarity_search(question), question=question)

    return results
