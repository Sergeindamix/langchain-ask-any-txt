import streamlit as st
import requests
import IPython
import soundfile as sf

def play_audio(audio):
    sf.write("speech_converted.wav", audio.numpy(), samplerate=16000)
    return IPython.display.Audio("speech_converted.wav")

def text_downloader(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return ""

def text_reader(text):
    audio = "speech_converted.wav"
    play_audio(audio)

def promptx(text):
    st.title("Text to Audio Converter")

    # Get the URL from the user
    url = st.text_input("Enter the URL of the text", "https://gamma.app/docs/sgx5tfyh0wttucu")

    if st.button("Download Text"):
        text = text_downloader(url)
        st.write(f"The text is {text}")

    if st.button("Generate Audio"):
        audio = text_reader(text)
        # Add your code here to play or save the audio
        st.audio(audio, format='audio/wav')
        
if __name__ == "__main__":
    promptx()
