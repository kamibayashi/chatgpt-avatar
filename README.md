# Voice Chat With ChatGPT

You can voice chat with ChatGPT.

1. save your conversation as a wav file.
2. convert wav file to text with whisper API.
3. send text to chatgpt.
4. playback reply from chatgpt on VOICEVOX.

# How to use

・Download VOICEVOX from here for speak avatar.

https://voicevox.hiroshiba.jp/

```sh
poetry install

export OPENAI_API_KEY="your api key"

poetry run python chatgpt_avatar/chatgpt_avatar.py
```

