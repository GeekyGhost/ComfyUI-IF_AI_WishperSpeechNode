import os
import datetime
import torch
import librosa
import textwrap
import torchaudio
from nltk.tokenize import sent_tokenize
import re
import scipy.io.wavfile as wav
from whisperspeech.pipeline import Pipeline

import nltk
nltk.download('punkt', quiet=True)

class IFWhisperSpeech:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": cls.sample_text()}),
                "file_name": ("STRING", {"default": "IF_whisper_speech"}),
                "speaker": (cls.get_audio_files(), {}),
                "torch_compile": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "cps": ("FLOAT", {"default": 14.0, "min": 10.0, "max": 20.0, "step": 0.25}),
                "overlap": ("INT", {"default": 100, "min": 0, "max": 200, "step": 1}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audios", "wav_16k_path")
    FUNCTION = "generate_audio"
    CATEGORY = "ImpactFrames💥🎞️"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, text, file_name, speaker, torch_compile, cps, overlap):
        return True

    @staticmethod
    def sample_text():
        return textwrap.dedent("""\
            Electromagnetism is a fundamental force of nature that encompasses the interaction between
            electrically charged particles. It is described by Maxwell's equations, which unify electricity, magnetism,
            and light into a single theory. In essence, electric charges produce electric fields that exert forces on
            other charges, while moving charges (currents) generate magnetic fields. These magnetic fields, in turn,
            can affect the motion of charges and currents. The interaction between electric and magnetic fields propagates
            through space as electromagnetic waves, which include visible light, radio waves, and X-rays. Electromagnetic
            forces are responsible for practically all the phenomena encountered in daily life, excluding gravity.
            """)

    @staticmethod
    def get_audio_files():
        audio_dir = os.path.join(os.path.dirname(__file__), "whisperspeech", "audio")
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".ogg")]
        return ["None"] + audio_files

    @staticmethod
    def split_and_prepare_text(text, cps):
        sentences = sent_tokenize(text)
        chunks = []
        chunk = ""
        for sentence in sentences:
            sentence = re.sub(r'[()]', ",", sentence.strip())
            sentence = re.sub(r',+', ",", sentence)
            sentence = re.sub(r'"+', "", sentence)
            sentence = re.sub(r'/', "", sentence)
            if len(chunk) + len(sentence) < 20 * cps:
                chunk += " " + sentence
            else:
                chunks.append(chunk.strip())
                chunk = sentence
        if chunk:
            chunks.append(chunk.strip())
        return chunks

    def generate_audio(self, text, file_name, speaker, torch_compile, cps, overlap):
        pipe = Pipeline(torch_compile=torch_compile)
        chunks = self.split_and_prepare_text(text, cps)

        if speaker != "None":
            speaker_file_path = os.path.join(os.path.dirname(__file__), "whisperspeech", "audio", speaker)
            speaker = pipe.extract_spk_emb(speaker_file_path)
        else:
            speaker = pipe.default_speaker

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_name = f"{file_name}_{timestamp}"
        comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(comfy_dir, "output", output_name)
        os.makedirs(output_dir, exist_ok=True)
        wav_16k_path = os.path.join(output_dir, f"{output_name}.wav")
        tmp_dir = os.path.join(comfy_dir, "temp", output_name)
        os.makedirs(tmp_dir, exist_ok=True)
        wav_temp_path = os.path.join(tmp_dir, f"{output_name}_temp.wav")

        r = []
        old_stoks = None
        old_atoks = None
        for chunk in chunks:
            print(chunk)
            stoks = pipe.t2s.generate(chunk, cps=cps, show_progress_bar=False)[0]
            stoks = stoks[stoks != 512]
            if old_stoks is not None:
                assert len(stoks) < 750 - overlap
                stoks = torch.cat([old_stoks[-overlap:], stoks])
                atoks_prompt = old_atoks[:, :, -overlap * 3:]
            else:
                atoks_prompt = None
            atoks = pipe.s2a.generate(stoks, atoks_prompt=atoks_prompt, speakers=speaker.unsqueeze(0), show_progress_bar=False)
            if atoks_prompt is not None:
                atoks = atoks[:, :, overlap * 3 + 1:]
            r.append(atoks)
            old_stoks = stoks
            old_atoks = atoks
            pipe.vocoder.decode_to_notebook(atoks)

        audios = []
        for i, atoks in enumerate(r):
            if i != 0:
                audios.append(torch.zeros((1, int(24000 * 0.5)), dtype=atoks.dtype, device=atoks.device))
            audios.append(pipe.vocoder.decode(atoks))

        torchaudio.save(wav_temp_path, torch.cat(audios, -1).cpu(), 24000)

        # Load and resample the audio to 16kHz
        audio, sr = librosa.load(wav_temp_path, sr=24000)
        resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        wav.write(wav_16k_path, rate=16000, data=(resampled_audio * 32767).astype('int16'))

        return audios, wav_16k_path

NODE_CLASS_MAPPINGS = {"IF_WhisperSpeech": IFWhisperSpeech}
NODE_DISPLAY_NAME_MAPPINGS = {"IF_WhisperSpeech": "IF Whisper Speech🌬️"}
