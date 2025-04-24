#!/usr/bin/env python3
# audio parameters tester

import os
import sys
import logging
import torchaudio
import librosa
import numpy as np
from dotenv import load_dotenv
from config import AudioConfig
from ui.core import Audio2MDUI

load_dotenv()

Config = AudioConfig()

def load_audio(file_path: str, target_sr=Config.TARGET_SAMPLE_RATE):
    """
    Load an audio file using torchaudio, convert to mono if necessary, and resample to target_sr.
    Returns a 1D numpy array and the sample rate.
    """
    try:
        waveform, sr = torchaudio.load(file_path)
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None, None
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0).numpy(), target_sr

def compute_audio_attributes(file_path: str):
    """
    Compute key audio attributes:
      - Duration (seconds)
      - Mean RMS energy (proxy for loudness/SNR)
      - Mean Zero Crossing Rate (noisiness)
      - Mean Spectral Centroid (brightness)
    Returns a dictionary of attribute values.
    """
    y, sr = load_audio(file_path)
    if y is None:
        return {}
    duration = len(y) / sr
    rms = librosa.feature.rms(y=y.astype(np.float32))
    mean_rms = float(np.mean(rms))
    zcr = librosa.feature.zero_crossing_rate(y.astype(np.float32))
    mean_zcr = float(np.mean(zcr))
    centroid = librosa.feature.spectral_centroid(y=y.astype(np.float32), sr=sr)
    mean_centroid = float(np.mean(centroid))
    return {
        "duration": duration,
        "mean_rms": mean_rms,
        "mean_zcr": mean_zcr,
        "mean_centroid": mean_centroid
    }

def dynamic_recommendations(attrs):
    """
    Based on computed audio attributes, dynamically recommend ideal decoding parameters.
    (These heuristics are illustrative. Adjust thresholds and values as needed.)
    """
    recommendations = {}
    # Recommend TRANSCRIBE_NUM_BEAMS:
    # If audio is quiet (low RMS) or noisy (high ZCR), a higher beam count may help.
    if attrs["mean_rms"] < 0.02 or attrs["mean_zcr"] > 0.1:
        recommendations["TRANSCRIBE_NUM_BEAMS"] = 7
    else:
        recommendations["TRANSCRIBE_NUM_BEAMS"] = 5

    # Recommend TRANSCRIBE_MAX_LENGTH:
    # For longer audio (full audiobook) you might want a higher max length.
    if attrs["duration"] < 1800:
        recommendations["TRANSCRIBE_MAX_LENGTH"] = 448
    else:
        recommendations["TRANSCRIBE_MAX_LENGTH"] = 512

    return recommendations

def print_diagnostics(file_path: str):
    """
    Compute audio attributes, compare them with dynamic recommendations, and report
    discrepancies with .env-defined parameters using rich Panels.
    """
    ui = Audio2MDUI()
    console = ui.get_console()
    
    # Retrieve overrides from .env using the helper function.
    env_max_length = Config.get_env("TRANSCRIBE_MAX_LENGTH", default=None, var_type=int)
    env_num_beams = Config.get_env("TRANSCRIBE_NUM_BEAMS", default=None, var_type=int)

    attrs = compute_audio_attributes(file_path)
    if not attrs:
        ui.error_panel(f"Failed to compute audio attributes for {file_path}.")
        return # Return instead of exiting

    recs = dynamic_recommendations(attrs)

    # Audio Attributes Panel
    audio_attrs_text = ui.create_text()
    audio_attrs_text.append(f"File: {file_path}\n")
    audio_attrs_text.append(f"Duration: {attrs['duration']:.2f} seconds\n")
    audio_attrs_text.append(f"Mean RMS Energy: {attrs['mean_rms']:.5f}\n")
    audio_attrs_text.append(f"Mean Zero Crossing Rate: {attrs['mean_zcr']:.5f}\n")
    audio_attrs_text.append(f"Mean Spectral Centroid: {attrs['mean_centroid']:.2f} Hz")
    ui.get_console().print(ui.create_panel(audio_attrs_text, title="Audio Attributes", border_style="blue"))

    # Recommended Parameters Panel
    recs_text = ui.create_text()
    recs_text.append(f"TRANSCRIBE_NUM_BEAMS: {recs['TRANSCRIBE_NUM_BEAMS']}\n")
    recs_text.append(f"TRANSCRIBE_MAX_LENGTH: {recs['TRANSCRIBE_MAX_LENGTH']}")
    ui.get_console().print(ui.create_panel(recs_text, title="Recommended Decoding Parameters (Dynamic)", border_style="green"))

    # .env Configuration Panel
    env_text = ui.create_text()
    if env_num_beams is not None:
        env_text.append(f"TRANSCRIBE_NUM_BEAMS is set in .env to: {env_num_beams}\n")
    else:
        env_text.append("TRANSCRIBE_NUM_BEAMS is not defined in .env.\n")
    if env_max_length is not None:
        env_text.append(f"TRANSCRIBE_MAX_LENGTH is set in .env to: {env_max_length}")
    else:
        env_text.append("TRANSCRIBE_MAX_LENGTH is not defined in .env.")
    ui.get_console().print(ui.create_panel(env_text, title=".env Configuration Overrides", border_style="yellow"))

    # Effective Configuration Panel
    effective_beams = env_num_beams if env_num_beams is not None else Config.TRANSCRIBE_NUM_BEAMS
    effective_max_length = env_max_length if env_max_length is not None else Config.TRANSCRIBE_MAX_LENGTH
    effective_text = ui.create_text()
    effective_text.append(f"Effective TRANSCRIBE_NUM_BEAMS: {effective_beams} (Using {'.env' if env_num_beams is not None else 'default'})\n")
    effective_text.append(f"Effective TRANSCRIBE_MAX_LENGTH: {effective_max_length} (Using {'.env' if env_max_length is not None else 'default'})")
    ui.get_console().print(ui.create_panel(effective_text, title="Effective Configuration", border_style="magenta"))

    # Recommendations Panel
    recommendations_text = ui.create_text()
    recommendations_made = False

    if effective_beams != recs["TRANSCRIBE_NUM_BEAMS"]:
        recommendations_made = True
        rec_text = ui.create_text()
        rec_text.append("Recommendation: Dynamic TRANSCRIBE_NUM_BEAMS is ")
        rec_text.append(str(recs['TRANSCRIBE_NUM_BEAMS']), style="bold green")
        rec_text.append(", effective is ")
        rec_text.append(str(effective_beams), style="bold yellow")
        rec_text.append(".\n")
        recommendations_text.append(rec_text)

        advice_text = ui.create_text()
        if env_num_beams is not None:
            advice_text.append("  Consider updating the value in .env.\n")
        else:
            advice_text.append(f"  Consider setting TRANSCRIBE_NUM_BEAMS={recs['TRANSCRIBE_NUM_BEAMS']} in .env.\n")
        recommendations_text.append(advice_text)

    if effective_max_length != recs["TRANSCRIBE_MAX_LENGTH"]:
        recommendations_made = True
        rec_text = ui.create_text()
        rec_text.append("Recommendation: Dynamic TRANSCRIBE_MAX_LENGTH is ")
        rec_text.append(str(recs['TRANSCRIBE_MAX_LENGTH']), style="bold green")
        rec_text.append(", effective is ")
        rec_text.append(str(effective_max_length), style="bold yellow")
        rec_text.append(".\n")
        recommendations_text.append(rec_text)

        advice_text = ui.create_text()
        if env_max_length is not None:
            advice_text.append("  Consider updating the value in .env.\n")
        else:
            advice_text.append(f"  Consider setting TRANSCRIBE_MAX_LENGTH={recs['TRANSCRIBE_MAX_LENGTH']} in .env.\n")
        recommendations_text.append(advice_text)

    if not recommendations_made:
        recommendations_text.append("Effective settings match dynamic recommendations.", style="bold green")
    else:
        recommendations_text.append("Note: If the current effective settings yield acceptable quality, you may keep them.", style="italic dim")

    ui.get_console().print(ui.create_panel(recommendations_text, title="Recommendations & Notes", border_style="cyan"))
    ui.get_console().print() # Add a blank line for separation between files

def process_audio_file(file_path: str):
    if not os.path.isfile(file_path):
        logging.error(f"Skipping {file_path} as it is not a file.")
        return
    
    logging.info(f"Processing file: {file_path}")
    print_diagnostics(file_path)

def is_audio_file(file_path: str) -> bool:
    try:
        torchaudio.info(file_path)
        return True
    except Exception as e:
        logging.debug(f"File {file_path} is not a valid audio file: {e}")
        return False

def main():
    ui = Audio2MDUI()
    if len(sys.argv) != 2:
        ui.error_panel(f"Usage: python {os.path.basename(sys.argv[0])} <audio_file or directory>")
        sys.exit(1)
    
    
    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        ui.error_panel(f"Input path does not exist: {input_path}")
        sys.exit(1)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[ui.configure_basic_logging()]
    )
    
    if os.path.isfile(input_path):
        if is_audio_file(input_path):
            process_audio_file(input_path)
        else:
            logging.error(f"File {input_path} is not a valid audio file.")
            sys.exit(1)
    elif os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            for file in files:
                file_path = os.path.join(root, file)
                if is_audio_file(file_path):
                    process_audio_file(file_path)
    else:
        logging.error(f"Invalid input: {input_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
