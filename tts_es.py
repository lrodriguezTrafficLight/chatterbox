"""Generador de audio TTS en español usando ChatterboxMultilingualTTS."""

import re
import torch
import torchaudio as ta
from pathlib import Path
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("output")
MAX_CHARS = 300  # Limite seguro por fragmento para no exceder max_text_tokens


def load_model():
    print(f"Cargando modelo en {DEVICE}...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
    print(f"Modelo cargado. GPU: {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")
    return model


def split_text(text, max_chars=MAX_CHARS):
    """Divide texto largo en fragmentos respetando oraciones completas."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip() if current else sentence
        else:
            if current:
                chunks.append(current)
            # Si una oracion sola excede el limite, dividir por comas
            if len(sentence) > max_chars:
                parts = sentence.split(", ")
                sub = ""
                for part in parts:
                    if len(sub) + len(part) + 2 <= max_chars:
                        sub = f"{sub}, {part}".strip(", ") if sub else part
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = part
                if sub:
                    chunks.append(sub)
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


def generate(model, text, filename):
    chunks = split_text(text)

    if len(chunks) == 1:
        print(f"Generando: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
        wav = model.generate(text, language_id="es")
    else:
        print(f"Texto largo detectado ({len(text)} chars). Dividiendo en {len(chunks)} fragmentos...")
        wavs = []
        # Silencio de 0.4s entre fragmentos
        silence = torch.zeros(1, int(model.sr * 0.4))
        for i, chunk in enumerate(chunks, 1):
            print(f"  [{i}/{len(chunks)}] \"{chunk[:60]}...\"")
            wav_chunk = model.generate(chunk, language_id="es")
            wavs.append(wav_chunk)
            if i < len(chunks):
                wavs.append(silence)
        wav = torch.cat(wavs, dim=1)

    output_path = OUTPUT_DIR / filename
    ta.save(str(output_path), wav, model.sr)
    duration = wav.shape[1] / model.sr
    print(f"Guardado en {output_path} ({duration:.1f}s)")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    model = load_model()
    count = 1

    print("\n--- TTS Español ---")
    print("Escribe el texto y presiona Enter para generar audio.")
    print("Soporta textos largos (se dividen automaticamente).")
    print("Escribe 'salir' para terminar.\n")

    while True:
        try:
            text = input("Texto: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not text or text.lower() == "salir":
            break

        filename = f"audio_{count:03d}.wav"
        generate(model, text, filename)
        count += 1
        print()

    print("Fin.")


if __name__ == "__main__":
    main()
