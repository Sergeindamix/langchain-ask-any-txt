import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings


def embds(text):
    st.title("Language Embeddings with Hugging Face")
    model_type = st.selectbox("Select a model type", ["HuggingFace", "HuggingFace with Instructor"])

    if model_type == "HuggingFace":
        model_name = st.text_input("Enter a Hugging Face model name", value="sentence-transformers/all-mpnet-base-v2")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

    else:
        model_name = st.text_input("Enter an Instructor model name", value="hkunlp/instructor-large")
        embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)

    text = st.text_input("Enter some text to embed")

    if st.button("Embed"):
        if not text:
            st.error("Please enter some text.")
            return

        if isinstance(embeddings, HuggingFaceEmbeddings):
            results = embeddings.embed_documents([text])[0]
        else:
            results = embeddings.embed_query(text)

        st.write("Embeddings:")
        st.json(results)


if __name__ == "__main__":
    embds()
