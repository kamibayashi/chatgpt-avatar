import json
import os
import time
import wave
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import openai
import pyaudio
import requests
import whisper
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms import OpenAIChat
from pydub import AudioSegment, playback

openai.api_key = os.environ["OPENAI_API_KEY"]


class ChatGPTAvatar:
    def __init__(self) -> None:
        # setting audio
        self.audio = pyaudio.PyAudio()
        self.channels: int = 1
        self.rate: int = 44100
        self.chunk: int = 2**11
        self.format: int = pyaudio.paInt16
        self.stream: pyaudio.PyAudio.Stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self.callback,
        )
        self.threshold: int = 4000
        self.time_after_speaking: int = 3
        self.status: int = 0
        self.output: Optional[wave.Wave_write] = None
        self.socket: Optional[str] = None

        # setting chatgpt
        prompt = PromptTemplate(
            input_variables=["history", "input"], template=self.__prompt_template()
        )

        llm = OpenAIChat(
            model_name="gpt-3.5-turbo",
            temperature=0,
            max_tokens=2000,
            prefix_messages=[{"role": "system", "content": self.__base_charactor()}],
        )

        self.conversation = ConversationChain(
            llm=llm,
            prompt=prompt,
            memory=ConversationBufferWindowMemory(k=4, memory_key="history"),
            verbose=True,
        )

    def __prompt_template(self) -> str:
        return """以下は、全知全能の全てを知っているAIのあなたとHumanが会話しています。

{history}
Human: {input}
AI:"""

    def __base_charactor(self) -> str:
        return """あなたは、親切で、創造的で、賢く、とてもフレンドリーで役に立つアシスタントです。
AIは質問に対する答えを知らない場合、正直に「知らない」と答えます。"""

    def callback(
        self,
        in_data: Any,
        frame_count: int,
        time_info: Dict[Any, Any],
        status: int,
    ) -> tuple[Any, Any]:
        if self.output is not None:
            self.output.writeframes(in_data)
        amp = np.frombuffer(in_data, np.int16)
        self.record(amp)
        out_data = in_data
        return (out_data, pyaudio.paContinue)

    def record(self, amp: np.ndarray) -> None:
        if (self.status == 0) and (amp.max() > self.threshold):
            self.status = 1
            self.open_file(datetime.now().strftime("%Y%m%d_%H%M%S"))
            self.start_at = time.time()
            print("Start Record.")
        elif (self.status == 1) and (amp.max() > self.threshold):
            self.start_at = time.time()
        elif (self.status == 1) and (amp.max() <= self.threshold):
            if (time.time() - self.start_at) > self.time_after_speaking:
                self.status = 0
                self.close_file()
                print("Finish Record.")
                self.call_chatgpt()

    def call_chatgpt(self) -> None:
        reply_message = self.conversation.predict(input=self.voice_to_text())
        self.talk_with_avatar(reply_message)

    def open_file(self, time_stamp: str) -> None:
        self.socket = f"./recorded/{time_stamp}_audio.wav"
        self.output = wave.open(self.socket, "wb")
        self.output.setnchannels(self.channels)
        self.output.setsampwidth(2)
        self.output.setframerate(self.rate)

    def close_file(self) -> None:
        if self.output is not None:
            self.output.close()
        self.output = None

    def close(self) -> None:
        self.audio.terminate()

    def voice_to_text(self, model_name: str = "medium") -> Any:
        model = whisper.load_model(model_name)
        result = model.transcribe(self.socket, fp16=False, verbose=True)
        return result["text"]

    def talk_with_avatar(
        self,
        text: str,
        speaker_id: int = 0,
        speed: float = 1.0,
        pitch: int = 0,
        volume: float = 1.0,
        intonation: float = 1.0,
    ) -> None:
        # query to VOICEVOX
        response = requests.post(
            "http://127.0.0.1:50021/audio_query",
            timeout=10,
            params={"text": text, "speaker": speaker_id},
        )

        query = response.json()

        # setting parameter
        query["speedScale"] = speed
        query["pitchScale"] = pitch
        query["volumeScale"] = volume
        query["intonationScale"] = intonation
        query["prePhonemeLength"] = 1
        query["postPhonemeLength"] = 1

        # query to VOICEVOX
        response = requests.post(
            "http://127.0.0.1:50021/synthesis",
            timeout=10,
            params={
                "speaker": speaker_id,
                "speed": "1.0",
                "enable_interrogative_upspeak": "true",
            },
            data=json.dumps(query),
        )

        # play audio
        playback.play(AudioSegment(response.content))


def main() -> None:
    while True:
        bot = ChatGPTAvatar()

        bot.stream.start_stream()
        while bot.stream.is_active():
            try:
                time.sleep(0.1)
            except KeyboardInterrupt:
                break

        bot.stream.stop_stream()
        bot.stream.close()
        bot.close()


if __name__ == "__main__":
    main()
