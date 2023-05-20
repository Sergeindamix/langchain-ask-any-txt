import sys
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

def run_flan_t5(line):
    if line:
        model_name = 'google/flan-t5-base'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        config = GenerationConfig(max_new_tokens=200)

        tokens = tokenizer(line, return_tensors="pt")
        outputs = model.generate(**tokens, generation_config=config)
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        st.write("Generated Text:")
        st.write(result)

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [Transformers](https://huggingface.co/transformers/)
    - LLM model from [Google's Flax community](https://huggingface.co/google)
    ''')

# Main content
st.header('LLM Text Generation')


