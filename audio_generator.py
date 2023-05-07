import requests
from io import BytesIO
from pydub import AudioSegment

def generate_audio(text, voice_id, api_key):
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

    headers = {
        "Accept": "audio/mp3",
        "xi-api-key": api_key
    }

    data = {
        "text": text,
        "voice_settings": {
            "speed": 1
        }
    }

    response = requests.post(tts_url, json=data, headers=headers, stream=True)

    if response.status_code == 200:
        # convert the streamed mp3 data to a playable audio format
        audio_data = BytesIO(response.content)
        audio_segment = AudioSegment.from_file(audio_data, format="mp3")
        return audio_segment
    else:
        print(f"Error generating audio. Status code: {response.status_code}")
        return None
