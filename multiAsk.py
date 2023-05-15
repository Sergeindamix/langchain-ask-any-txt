import streamlit as st

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
