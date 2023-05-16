import streamlit as st
from langchain.model_laboratory import ModelLaboratory
from langchain.prompts import PromptTemplate
overal_temperature = 0.1
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import os
os.environ['OPENAI_API_KEY'] = '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'
os.environ["COHERE_API_KEY"] = "RHHr7yYclxRtRF5Pt0xmK4sQsacefaAe3n5EnXTq" #"iGcx36mfWxrjgSEIAq22FklWNovaeqG18xmaxu3n"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
def run_comparison(question):
    overal_temperature = 0.1
    flan_20B = HuggingFaceHub(repo_id="google/flan-ul2",
                              model_kwargs={"temperature": overal_temperature,
                                            "max_new_tokens": 200}
                              )
    flan_t5xxl = HuggingFaceHub(repo_id="google/flan-t5-xxl",
                                model_kwargs={"temperature": overal_temperature,
                                              "max_new_tokens": 200}
                                )
    gpt_j6B = HuggingFaceHub(repo_id="EleutherAI/gpt-j-6B",
                             model_kwargs={"temperature": overal_temperature,
                                           "max_new_tokens": 100}
                             )
    from langchain.llms import OpenAI, OpenAIChat

    chatGPT_turbo = OpenAIChat(model_name='gpt-3.5-turbo',
                               temperature=overal_temperature,
                               max_tokens=256,
                               )

    gpt3_davinici_003 = OpenAI(model_name='text-davinci-003',
                               temperature=overal_temperature,
                               max_tokens=256,
                               )
    from langchain.llms import Cohere
    cohere_command_xl = Cohere(model='command-xlarge',
                               temperature=0.1,
                               max_tokens=256)
    cohere_command_xl_nightly = Cohere(model='command-xlarge-nightly',
                                       temperature=0.1,
                                       max_tokens=256)
    template = """Question: {question}

    Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    lab = ModelLaboratory.from_llms([
        flan_20B,
        cohere_command_xl,
        cohere_command_xl_nightly
    ], prompt=prompt)

    # Obtener los resultados de la comparación
    results = lab.compare(question)

    # Verificar si los resultados son None antes de iterar
    if results is not None:
        # Mostrar los resultados en Streamlit
        for result in results:
            model_name = result['model_name']
            response = result['response']

            # Mostrar el nombre del modelo y la respuesta generada
            st.subheader(f"Modelo: {model_name}")
            st.write(f"Respuesta: {response}")
            st.write("---")  # Separador entre cada respuesta
    else:
        st.write("No se encontraron resultados.")
