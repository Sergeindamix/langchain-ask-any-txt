import streamlit as st
from langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Crear una función para ejecutar el modelo y generar respuestas
def run_llm_model(prompt):
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_file='llama-2-7b-chat.ggmlv3.q2_K.bin',
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    template = """
    [INST] <<SYS>>
    You are a helpful, respectful, and honest assistant. Your answers are always brief.
    <</SYS>>
    {text}[/INST]
    """

    prompt_template = PromptTemplate(template=template, input_variables=["text"])

    llm_chain = LLMChain(prompt=prompt_template, llm=llm)

    response = llm_chain.run(prompt)
    return response

# Crear la aplicación Streamlit
def main():
    st.title("LLM Chat")

    # Agregar un área de texto para ingresar la pregunta
    user_input = st.text_area("Ingrese su pregunta:")

    if st.button("Obtener respuesta"):
        if user_input:
            # Ejecutar el modelo y obtener la respuesta
            response = run_llm_model(user_input)

            # Mostrar la respuesta
            st.subheader("Respuesta:")
            st.write(response)
        else:
            st.warning("Por favor, ingrese una pregunta.")

if __name__ == "__main__":
    main()
