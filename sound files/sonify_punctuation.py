"""
PUNCTUATION SONIFICATION
========================
Converts punctuation positional data from Claude's conversation
into audio files — one WAV per mark.

Each occurrence of a mark becomes a sound event at the
corresponding moment in a 20-second audio piece.
The clustering and rhythm of the machine's usage becomes the sound.

REQUIREMENTS
------------
pip install numpy scipy

USAGE
-----
1. Place this script in the same folder as punctuation_data.csv
2. Run: python sonify_punctuation.py
3. Six WAV files will be generated, one per mark

SOUND DESIGN
------------
Each mark has a distinct tone character reflecting its function:
  Full Stop    — low sine tone, hard cutoff (440Hz → 220Hz)
  Comma        — mid sine tone, soft fade (330Hz)
  Semicolon    — two-tone pulse (275Hz + 330Hz)
  Colon        — clean mid tone, short (440Hz)
  Question     — rising tone (220Hz → 440Hz)
  Exclamation  — sharp high tone (880Hz)
"""

import numpy as np
import csv
import os
from scipy.io import wavfile

# ─────────────────────────────────────────────
# SETTINGS — adjust these to change the output
# ─────────────────────────────────────────────

DURATION_SECONDS = 30       # Length of each audio piece
SAMPLE_RATE = 44100          # Standard audio quality
OUTPUT_FOLDER = "sounds"     # Folder to save WAV files

# Tone duration per event (seconds)
# Shorter = more percussive, longer = more sustained
TONE_DURATIONS = {
    "full_stop":     0.06,
    "comma":         0.04,
    "semicolon":     0.05,
    "colon":         0.03,
    "question_mark": 0.12,
    "exclamation":   0.04,
}

# Volume per mark (0.0 to 1.0)
VOLUMES = {
    "full_stop":     0.7,
    "comma":         0.45,
    "semicolon":     0.5,
    "colon":         0.4,
    "question_mark": 0.8,
    "exclamation":   0.9,
}

# Base frequency per mark (Hz)
FREQUENCIES = {
    "full_stop":     220,
    "comma":         330,
    "semicolon":     275,
    "colon":         440,
    "question_mark": 220,
    "exclamation":   880,
}


# ─────────────────────────────────────────────
# TONE GENERATORS
# ─────────────────────────────────────────────

def sine_tone(freq, duration, volume, sample_rate, envelope="hard"):
    """Generate a sine wave tone with optional envelope."""
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    wave = np.sin(2 * np.pi * freq * t) * volume

    if envelope == "fade":
        # Soft fade out
        fade = np.linspace(1.0, 0.0, n_samples)
        wave *= fade
    elif envelope == "hard":
        # Hard cutoff — just as it is
        pass
    elif envelope == "click":
        # Very short attack, fast decay
        attack = int(n_samples * 0.05)
        decay = n_samples - attack
        env = np.concatenate([
            np.linspace(0, 1, attack),
            np.linspace(1, 0, decay) ** 2
        ])
        wave *= env

    return wave


def rising_tone(freq_start, freq_end, duration, volume, sample_rate):
    """Rising frequency tone for question mark."""
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    freqs = np.linspace(freq_start, freq_end, n_samples)
    wave = np.sin(2 * np.pi * np.cumsum(freqs) / sample_rate) * volume
    # Apply fade envelope
    fade = np.linspace(1.0, 0.0, n_samples)
    wave *= fade
    return wave


def double_tone(freq1, freq2, duration, volume, sample_rate):
    """Two-frequency tone for semicolon."""
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    wave1 = np.sin(2 * np.pi * freq1 * t) * volume * 0.6
    wave2 = np.sin(2 * np.pi * freq2 * t) * volume * 0.4
    wave = wave1 + wave2
    fade = np.linspace(1.0, 0.0, n_samples)
    wave *= fade
    return wave


def generate_tone(mark_name, duration, volume, freq, sample_rate):
    """Generate appropriate tone for each mark type."""
    if mark_name == "full_stop":
        return sine_tone(freq, duration, volume, sample_rate, envelope="hard")
    elif mark_name == "comma":
        return sine_tone(freq, duration, volume, sample_rate, envelope="fade")
    elif mark_name == "semicolon":
        return double_tone(freq, freq * 1.2, duration, volume, sample_rate)
    elif mark_name == "colon":
        return sine_tone(freq, duration, volume, sample_rate, envelope="click")
    elif mark_name == "question_mark":
        return rising_tone(freq, freq * 2, duration, volume, sample_rate)
    elif mark_name == "exclamation":
        return sine_tone(freq, duration, volume, sample_rate, envelope="click")
    else:
        return sine_tone(freq, duration, volume, sample_rate, envelope="fade")


# ─────────────────────────────────────────────
# MAIN SONIFICATION
# ─────────────────────────────────────────────

def load_data(csv_path):
    """Load punctuation events from CSV."""
    events = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mark = row['mark']
            time_fraction = float(row['time_fraction'])
            if mark not in events:
                events[mark] = []
            events[mark].append(time_fraction)
    return events


def sonify_mark(mark_name, time_fractions, output_path):
    """Convert a list of time positions into a WAV audio file."""

    total_samples = int(SAMPLE_RATE * DURATION_SECONDS)
    audio = np.zeros(total_samples, dtype=np.float32)

    freq = FREQUENCIES.get(mark_name, 440)
    volume = VOLUMES.get(mark_name, 0.5)
    tone_dur = TONE_DURATIONS.get(mark_name, 0.05)

    tone = generate_tone(mark_name, tone_dur, volume, freq, SAMPLE_RATE)
    tone_samples = len(tone)

    placed = 0
    for t_frac in time_fractions:
        # Map time_fraction (0.0–1.0) to sample position
        start = int(t_frac * total_samples)
        end = start + tone_samples

        if end > total_samples:
            end = total_samples
            tone_slice = tone[:end - start]
        else:
            tone_slice = tone

        audio[start:end] += tone_slice
        placed += 1

    # Normalise to prevent clipping
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    # Convert to 16-bit PCM
    audio_int = (audio * 32767).astype(np.int16)

    wavfile.write(output_path, SAMPLE_RATE, audio_int)
    return placed


def main():
    csv_path = "punctuation_data.csv"

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        print("Place punctuation_data.csv in the same folder as this script.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("Loading data...")
    events = load_data(csv_path)

    marks_to_process = [
        "full_stop",
        "comma",
        "semicolon",
        "colon",
        "question_mark",
        "exclamation",
    ]

    print(f"\nGenerating {DURATION_SECONDS}-second audio files...\n")
    print(f"{'Mark':<20} {'Events':>8}   Output")
    print("─" * 52)

    for mark in marks_to_process:
        if mark not in events:
            print(f"{mark:<20} {'0':>8}   (no data found)")
            continue

        time_fractions = events[mark]
        filename = f"{mark}.wav"
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        placed = sonify_mark(mark, time_fractions, output_path)
        print(f"{mark:<20} {placed:>8}   → {output_path}")

    print(f"\nDone. Files saved to '{OUTPUT_FOLDER}/' folder.")
    print("\nNote on the sounds:")
    print("  Colon      — densest, most structural (9,727 events)")
    print("  Semicolon  — second most frequent (6,021)")
    print("  Comma      — mid frequency (3,787)")
    print("  Full Stop  — sparse, considered (3,373)")
    print("  Question   — rare, rising tone (275)")
    print("  Exclamation— almost silent, 28 events only")


if __name__ == "__main__":
    main()
