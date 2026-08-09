import base64
import tempfile
import time
from pathlib import Path

import runpod
import torch
from qwen_tts import Qwen3TTSModel

MODEL_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

print(f"Loading {MODEL_REPO} on cuda:0 ...")
model = Qwen3TTSModel.from_pretrained(
    MODEL_REPO,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
print("Model loaded, worker ready.")


def handler(job):
    job_input = job["input"]
    text = job_input["text"]
    language = job_input.get("language", "English")
    ref_text = job_input["ref_text"]
    ref_audio_b64 = job_input["ref_audio_b64"]

    with tempfile.TemporaryDirectory() as tmp:
        ref_audio_path = Path(tmp) / "ref.wav"
        ref_audio_path.write_bytes(base64.b64decode(ref_audio_b64))

        start = time.time()
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=str(ref_audio_path),
            ref_text=ref_text,
        )
        elapsed = time.time() - start

        wav = wavs[0] if isinstance(wavs, list) else wavs
        out_path = Path(tmp) / "out.wav"
        import soundfile as sf

        sf.write(out_path, wav, sr)

        return {
            "audio_b64": base64.b64encode(out_path.read_bytes()).decode("ascii"),
            "sample_rate": sr,
            "seconds_taken": round(elapsed, 2),
        }


runpod.serverless.start({"handler": handler})
