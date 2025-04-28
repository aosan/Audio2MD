# Audio2MD: Audio-to-Markdown Private Transcriber

Get a private pipeline to transcribe audio from common audio and video file formats into Markdown text using the Hugging Face Whisper model. No data leaves your premise.
Optionally, the transcript can be refined using a private LLM (e.g., via Ollama or vLLM), or other external or public OpenAI API compatible providers.

## 🧰 What You Get:

🔒 On-Premise Whisper Transcription – No cloud, no leaks

✍️ Markdown Output – Ready to use, edit, or publish

⚙️ LLM Post-Processing (Optional) – Use private or public models

🧩 Flexible Integration – Works with your stack, your rules

## TL;DR: Quick Start

**Goal:** Get a Markdown transcript from audio/video files using local resources.

1. **Install:**

    Make sure you have Python 3.13+ and FFmpeg installed first, then install the dependencies:
    
    ```shell
    pip install -r requirements.txt
    ```

2. **Configure Environment Variables:**

    Create a `.env` file using the `.env.example` in the project directory and adjust the variables as needed.

3.  **Transcribe:**

    ```shell
    # For a single file (output saved as input_file_transcript.md)
    python audio2md.py input_file.mp3

    # Or for all files in a folder (output in new folder 'audio_files_transcripts_...')
    python audio2md.py path/to/your/audio_files/
    ```

    *   _(Optional)_ If you run out of GPU memory (OOM error), edit the `.env` file and set `TRANSCRIBE_MODEL_NAME` to a smaller model like `openai/whisper-medium` or `openai/whisper-base`.

4.  **Style (Optional):**

    ```shell
    # Make sure your LLM service (e.g., Ollama or vLLM) is running and configured in .env
    python a2md_orthography.py your_audio_or_video_transcript.md
    ```

_(See below for detailed configuration, features, and troubleshooting.)_

## Features

- **Automatic Speech Recognition (ASR):**  
  Transcribe audio using the Whisper model.
- **Audio Splitting:**  
  Optionally split long audio files using silence detection (via `librosa`).
- **LLM-based Transcript Refinement:**  
  When configured (`TRANSCRIPT=llm`), refine the transcript via a LLM service, private or public.
- **Configurable Decoding Settings:**  
  Fine-tune transcription quality, performance, and memory usage using `.env` variables (see Setup section).
- **Broad Format Support:** Processes common audio and video file types by extracting the audio stream (see Supported Formats).

## Requirements

- **Python:** 3.13+
- **Dependencies:** All required Python packages are listed in [requirements.txt](requirements.txt) (including `transformers`, `torch`, `openai`, etc.). Install them using:

  ```shell
  pip install -r requirements.txt
  ```

- **FFmpeg (Required for Broad Format Support):** Essential for handling most audio and video formats beyond basic WAV. `torchaudio` relies on `ffmpeg` being installed and available in your system PATH. Install it via your system's package manager (e.g., `pacman -S ffmpeg`, `dnf install ffmpeg`, `apt install ffmpeg`, `brew install ffmpeg`).

## Setup

Clone the repository or download a release package and navigate to its directory.

**Install Dependencies:**

```shell
pip install -r requirements.txt
```

**Configure Environment Variables:**
Create a `.env` file using the .env.example in the project directory and adjust the following variables as needed:

- **Shared Settings (Used by both `audio2md.py` and `a2md_orthography.py`):**
  - **`LLM_API_BASE_URL`**: Base URL for the LLM's OpenAI-compatible API endpoint. Examples: `http://localhost:11434/v1` (Ollama), `http://localhost:8000/v1` (vLLM default).
  - **`LLM_API_KEY`**: Optional API key if the LLM endpoint requires authentication (leave blank if not needed).
  - **`LLM_MODEL_NAME`**: Name/identifier of the LLM model served by the endpoint. **Important:** Must match the exact format expected by your specific LLM service (e.g., `mistral` for Ollama, `google/gemma-3-1b-it` for vLLM).
  - **`FRAGMENT_SIZE`**: Max character size for text chunks sent to the LLM (default: 8000).
  - **`CHUNK_OVERLAP`**: Overlap size between text chunks for context (default: 500). Must be smaller than `FRAGMENT_SIZE`.
  - **`LLM_TIMEOUT`**: Timeout in seconds for LLM API requests (passed to the OpenAI client, default: 120).
  - **`LLM_RETRIES`**: Max retry attempts for failed LLM API calls (passed to the OpenAI client, default: 3).
  - **`LLM_TEMPERATURE`**: LLM temperature setting (0.0-2.0). If set, passed in API request; if unset, the LLM service's default temperature is used.
  - **`TRANSCRIPT_PROMPT`**: Path to the prompt template file used for LLM refinement (default: `prompts/transcript_prompt.txt`). Used by `a2md_orthography.py` and by `audio2md.py` when `TRANSCRIPT_MODE=llm`.
  - **`TARGET_SAMPLE_RATE`**: Sample rate to resample audio to (default: 16000). Should match the Whisper model's requirement.
  - **`DEBUG`**: Set to `True` to enable detailed debug logging to `audio2md.log` or `a2md_orthography.log` (overwritten each run). Default: `False`.
  - **`PYTORCH_CUDA_ALLOC_CONF`**: Advanced PyTorch CUDA memory allocation setting (optional, e.g., `backend:cudaMallocAsync` or `max_split_size_mb:128`). Can help prevent fragmentation OOM errors.
  - **`CUDA_LAUNCH_BLOCKING`**: Set to `True` to enable CUDA_LAUNCH_BLOCKING for debugging CUDA errors. Default: `False`.
- **`audio2md.py` Specific Settings:**
  - **`TRANSCRIBE_MODEL_NAME`**: Whisper model to use (e.g., `openai/whisper-large-v3`, `openai/whisper-medium`). Smaller models use less VRAM but may have lower transcription quality.
  - **`CHUNK_LENGTH_S`**: Audio chunk length in seconds for ASR processing (default: 30). Affects VRAM usage.
  - **`AUDIO_BATCH_SIZE`**: Number of audio chunks processed in parallel by ASR (default: 4). Reduce if encountering OOM errors.
  - **`MAX_CHUNK_DURATION_S`**: Maximum duration (seconds) for audio segments when splitting (default: 600). Smaller values reduce memory per chunk but increase the number of chunks.
  - **`TRANSCRIBE_MAX_LENGTH`**: Maximum number of tokens generated per chunk by ASR (default: 448).
  - **`TRANSCRIBE_NUM_BEAMS`**: Number of beams for ASR decoding (default: 5). Higher values may improve transcription quality but slow down processing.
  - **`TRANSCRIBE_LANGUAGE`**: Target language for transcription (e.g., `en`, `es`, `fr`). Leave empty or set to `None` in `.env` to attempt auto-detection (may be less reliable than specifying).
  - **`RETRY_TRANSCRIBE`**: Number of retry attempts if a chunk transcription fails due to OOM (default: 2).
  - **`SPLIT_AUDIO`**: Set to `yes` (default) to split long audio files, `no` to process whole. Splitting is recommended for large files.
  - **`TRANSCRIPT_MODE`**: `direct` (default) for raw Whisper output, `llm` to refine using an LLM via an OpenAI-compatible API immediately after transcription.
  - **`DEVICE`**: Device selection for processing: 'auto', 'gpu', or 'cpu'. Default is 'auto', to detect and use the best available device.

## Supported Formats

The script attempts to process files with the following extensions by extracting their audio stream:

- **Audio:** `.mp3`, `.wav`, `.flac`, `.m4a`, `.m4b`, `.ogg`, `.opus`, `.aac`, `.wma`
- **Video (Audio Extraction):** `.mp4`, `.mkv`, `.avi`, `.mov`, `.wmv`

**Note:**
- Success depends on your `ffmpeg` installation supporting the specific codecs used with your file containers.

## Usage

This script supports three modes:

**Single File Mode:**

Process one audio file and produce one Markdown file.

```shell
python audio2md.py <input_audio_or_video_file> [<output_markdown_file>]
```

Example:

```shell
python audio2md.py meeting_recording.mp3 meeting_recording_transcript.md
```

**Folder Pair Mode:**

Process all supported audio/video files in an input folder and output Markdown files to a specified output folder.

```shell
python audio2md.py <input_folder> <output_folder>
```

Example:

```shell
python audio2md.py interviews_audio interviews_audio_transcript
```

**Single Folder Mode:**

Process all supported audio/video files in an input folder and automatically create an output folder with current timestamp (named `<input_folder>_transcripts_YYYYMMDD_HHMM`).

```shell
python audio2md.py <input_folder>
```

Example:

```shell
python audio2md.py interviews_audio
```

## Parameter Testing Utility `test_audio2md_parameters.py`

This utility script analyzes an audio file and provides recommendations for the `TRANSCRIBE_NUM_BEAMS` and `TRANSCRIBE_MAX_LENGTH` parameters based on basic audio characteristics (duration, loudness, noisiness). It compares these dynamic recommendations against the values currently set in your `.env` file (or the defaults used by `audio2md.py` if not set in `.env`).

**Purpose:**

- Get insights into the characteristics of your audio file.
- Receive heuristic-based suggestions for potentially optimizing transcription parameters (`TRANSCRIBE_NUM_BEAMS`, `TRANSCRIBE_MAX_LENGTH`).
- Easily compare recommended settings with your current configuration.

**Note:** The recommendations are based on simple heuristics and are intended as a starting point. The optimal settings may vary depending on the specific audio content and desired transcription quality.

**Usage:**

```shell
python test_audio2md_parameters.py <input_audio_or_video_file>
```

**Example:**

```shell
python test_audio2md_parameters.py meeting_recording.mp3
```

The script will output:
- Computed audio attributes (Duration, Mean RMS, Mean ZCR, Mean Spectral Centroid).
- Dynamically recommended values for `TRANSCRIBE_NUM_BEAMS` and `TRANSCRIBE_MAX_LENGTH`.
- The values currently configured via your `.env` file (if any).
- The *effective* settings that `audio2md.py` would use (considering `.env` overrides or defaults).
- Actionable recommendations if the effective settings differ from the dynamic recommendations.

## Troubleshooting

Common Issues and Solutions:

- **FFmpeg Not Found:**
  - Verify installation with `ffmpeg -version`
  - Install via package manager (apt/brew/choco) or from https://ffmpeg.org/

- **CUDA/GPU Issues:**
  - Check drivers and CUDA installation
  - Try reducing batch size or model size
  - Processing will fall back to CPU processing if GPU initialization fails

- **Out-of-Memory (OOM) Errors:** If you encounter CUDA OOM errors during transcription:
  - If doing transcription and LLM processing on the same machine, stop the LLM to release the VRAM, i.e. `ollama stop mistral`, prior to the transcription job
  - Reduce `AUDIO_BATCH_SIZE`.
  - Reduce `MAX_CHUNK_DURATION_S`.
  - Reduce `CHUNK_LENGTH_S`.
  - Use a smaller `TRANSCRIBE_MODEL_NAME`.
  - Adjust `PYTORCH_CUDA_ALLOC_CONF` (advanced).
  - Ensure `SPLIT_AUDIO=yes` is set for large files.
- **LLM Service Issues (Refinement/Styling):**
  - Check the `LLM_API_BASE_URL` points to the correct OpenAI-compatible endpoint (e.g., `http://localhost:11434/v1` for Ollama).
  - Ensure the LLM service is running.
  - Verify the `LLM_MODEL_NAME` is correct for the running service.
  - Check if an `LLM_API_KEY` is required and set correctly.
  - Review logs (e.g., `audio2md.log`, `a2md_orthography.log` if `DEBUG=True`) for specific API errors (authentication, connection, rate limits, etc.).
  - If the LLM service is slow, timeouts might occur (configure `LLM_TIMEOUT` and `LLM_RETRIES` in `.env`). Consider warming up the service beforehand.

## Fine-Tuning Decoding Settings

Adjust parameters like `TRANSCRIBE_NUM_BEAMS`, `CHUNK_LENGTH_S`, `AUDIO_BATCH_SIZE`, and `MAX_CHUNK_DURATION_S` in the `.env` file to balance transcription quality, processing speed, and memory usage.

## Transcript Styling `a2md_orthography.py`

This separate script allows you to apply LLM-based styling and formatting to *existing* Markdown transcript files generated by `audio2md.py` or other means. It sends the transcript to the configured LLM with a specific prompt (defined in `transcript_prompt.txt` by default), and saves the styled output to a new file or folder.

**Purpose:**

- Apply consistent formatting (e.g., paragraph breaks, speaker labels if prompted correctly).
- Make minor typing adjustments introduced by ASR.
- Structure the transcript based on the LLM prompt.

**Relationship with `audio2md.py`'s `TRANSCRIPT` Setting:**

Note: `a2md_orthography.py` operates independently of the `TRANSCRIPT` setting in the `.env` file (which only affects `audio2md.py`). You can use `a2md_orthography.py` to style transcripts whether they were initially generated by `audio2md.py` using `TRANSCRIPT=direct` (raw ASR output) or `TRANSCRIPT=llm` (ASR + initial LLM refinement). It can also be used on Markdown transcripts obtained from other sources.

**Usage:**

The script can process a single `.md` file or all `.md` files within a directory. Output is saved with a `_orthography_YYYYMMDD_HHMM` suffix to avoid overwriting originals.

*   **Single File Mode:**
    ```shell
    python a2md_orthography.py <input_transcript.md> [--prompt-file <custom_prompt.txt>]
    ```
    Example:
    ```shell
    python a2md_orthography.py meeting_transcript.md
    # Output: meeting_transcript_orthography_20250417_1130.md (timestamp varies)
    ```

*   **Directory Mode:**
    ```shell
    python a2md_orthography.py <input_folder_with_md_files> [--prompt-file <custom_prompt.txt>]
    ```
    Example:
    ```shell
    python a2md_orthography.py interviews_transcripts
    # Output: Creates folder interviews_transcripts_orthography_20250417_1130
    #         containing orthographically corrected versions of all .md files from the input folder.
    ```

*   **Arguments:**
    *   `<input_path>`: Path to the input `.md` file or directory containing `.md` files.
    *   `--prompt-file <path>`: (Optional) Specify a custom prompt template file. Defaults to `transcript_prompt.txt`.

**Configuration (`.env` variables used):**

`a2md_orthography.py` uses the shared settings defined in the `.env` file (see "Setup" section above), including:
- `LLM_API_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_NAME`
- `FRAGMENT_SIZE`, `CHUNK_OVERLAP`
- `LLM_TIMEOUT`, `LLM_RETRIES`, `LLM_TEMPERATURE`
- `TRANSCRIPT_PROMPT`
- `DEBUG`

## Experimenting with Prompts

The quality and nature of the output from `a2md_orthography.py` heavily depend on the instructions given to the LLM via the prompt template file, and the size or performance of the LLM. Try different prompts until you achieve the desired results.

-   **Default Prompt:** The script uses `transcript_prompt.txt` by default. You can edit this file directly to change the instructions for all subsequent runs.
-   **Custom Prompt File:** For more controlled experiments or different styling tasks, create a new text file (e.g., `my_custom_prompt.txt`) with your desired instructions. Use the `--prompt-file` argument to specify it:
    ```shell
    python a2md_orthography.py input.md --prompt-file my_custom_prompt.txt
    ```
-   **Tailoring Prompts:** Different LLMs (specified by `LLM_MODEL_NAME` in `.env`) respond differently to prompts. You may need to adjust your instructions based on the model you are using and the specific formatting or corrections you want (e.g., adding speaker labels, fixing specific punctuation, sections and paragraphs editing etc.). 

## License

GNU Affero General Public License v3

## Warranty

This code is provided as-is, with no warranty. Use at your own risk. It might eat your files, rename your dog, or set all your alarms to 3:17 AM. No promises, no guarantees.
