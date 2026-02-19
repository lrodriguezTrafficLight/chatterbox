"""Generador de audio TTS en español usando ChatterboxMultilingualTTS."""

import argparse
import re
import torch
import torchaudio as ta
from pathlib import Path
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("output")
MAX_CHARS = 300


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


def generate_paragraph_wav(model, para, index, total):
    """Genera el waveform de un parrafo individual (puede tener multiples chunks)."""
    chunks = split_text(para)
    if len(chunks) == 1:
        print(f"  [{index}/{total}] \"{para[:70]}...\"")
        return model.generate(para, language_id="es")
    else:
        print(f"  [{index}/{total}] Parrafo largo ({len(chunks)} partes): \"{para[:50]}...\"")
        wavs = []
        short_silence = torch.zeros(1, int(model.sr * 0.3))
        for j, chunk in enumerate(chunks, 1):
            print(f"    [{j}/{len(chunks)}] \"{chunk[:60]}...\"")
            wav = model.generate(chunk, language_id="es")
            wavs.append(wav)
            if j < len(chunks):
                wavs.append(short_silence)
        return torch.cat(wavs, dim=1)


def generate_from_paragraphs(model, paragraphs, filename, pause=0.8):
    """Genera un unico archivo de audio con todos los parrafos."""
    wavs = []
    silence = torch.zeros(1, int(model.sr * pause))
    total = len(paragraphs)

    for i, para in enumerate(paragraphs, 1):
        wav = generate_paragraph_wav(model, para, i, total)
        wavs.append(wav)
        if i < total:
            wavs.append(silence)

    final_wav = torch.cat(wavs, dim=1)
    output_path = OUTPUT_DIR / filename
    ta.save(str(output_path), final_wav, model.sr)
    duration = final_wav.shape[1] / model.sr
    print(f"\nGuardado en {output_path} ({duration:.1f}s, {total} parrafos)")


def generate_individual_files(model, paragraphs, base_name, pause=0.8):
    """Genera un archivo de audio individual por cada parrafo."""
    total = len(paragraphs)
    stem = Path(base_name).stem

    for i, para in enumerate(paragraphs, 1):
        wav = generate_paragraph_wav(model, para, i, total)
        filename = f"{stem}_{i:03d}.wav"
        output_path = OUTPUT_DIR / filename
        ta.save(str(output_path), wav, model.sr)
        duration = wav.shape[1] / model.sr
        print(f"    -> {output_path} ({duration:.1f}s)")

    print(f"\n{total} archivos generados en {OUTPUT_DIR}/")


def generate_single(model, text, filename):
    """Genera audio a partir de un texto, dividiendolo si es necesario."""
    chunks = split_text(text)

    if len(chunks) == 1:
        print(f"Generando: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
        wav = model.generate(text, language_id="es")
    else:
        print(f"Texto largo ({len(text)} chars, {len(chunks)} fragmentos)...")
        wavs = []
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
    parser = argparse.ArgumentParser(description="TTS en español con Chatterbox")
    parser.add_argument("-f", "--file", type=str,
                        help="Archivo de texto con parrafos separados por lineas en blanco")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Nombre del archivo de salida (default: audio_001.wav)")
    parser.add_argument("-p", "--pause", type=float, default=0.8,
                        help="Pausa en segundos entre parrafos (default: 0.8)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    model = load_model()

    if args.file:
        # Modo archivo: leer parrafos separados por lineas en blanco
        text = Path(args.file).read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        filename = args.output or "speech.wav"
        print(f"\n--- Archivo: {args.file} ({len(paragraphs)} parrafos) ---")
        print("  [1] Archivo unico (todos los parrafos en un solo .wav)")
        print(f"  [2] Archivos individuales (un .wav por parrafo)\n")

        choice = input("Selecciona modo (1/2): ").strip()
        print()

        if choice == "2":
            generate_individual_files(model, paragraphs, filename, pause=args.pause)
        else:
            generate_from_paragraphs(model, paragraphs, filename, pause=args.pause)
    else:
        # Modo interactivo
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

            filename = args.output or f"audio_{count:03d}.wav"
            generate_single(model, text, filename)
            count += 1
            print()

    print("Fin.")


if __name__ == "__main__":
    main()
