#!/usr/bin/env python3
# Main script for audio to markdown transcription with optional LLM refinement

from accelerate import load_checkpoint_and_dispatch, Accelerator
import os
import sys
import logging
import datetime
import multiprocessing
from dotenv import load_dotenv
import warnings
from ui.core import Audio2MDUI
from llm_api_client import LLMApiClient
from processing import determine_file_paths
from rich.progress import Progress
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.console import Console as RichConsole
from rich.align import Align
from collections import deque

# Initialize UI components early for potential pre-run messages
ui = Audio2MDUI()
console = ui.get_console()

# Load environment variables early since they affect later imports
load_dotenv()

# Conditional warning suppression for cleaner output in production
# Debug mode shows all warnings for troubleshooting
from config import TranscriptionConfig, AudioConfig
Config = TranscriptionConfig()
Audio_Config = AudioConfig()

is_debug_mode = Config.DEBUG # Use the instance attribute

# Conditionally suppress UserWarnings during transformers import
if not is_debug_mode:
    # Reduce logging level for transformers in non-debug mode
    logging.getLogger("transformers").setLevel(logging.ERROR)
    # Filter ALL UserWarnings globally (if not debug) to prevent console pollution
    warnings.filterwarnings("ignore", category=UserWarning)
    with warnings.catch_warnings(): # Keep this context for import-specific filtering if needed
        warnings.simplefilter("ignore", category=UserWarning) # Suppress during import (may be redundant now)
        # Import transformers inside the context manager
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
else:
    # Import normally if in debug mode
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from openai import APIConnectionError, AuthenticationError, RateLimitError, BadRequestError, APITimeoutError, APIError
import librosa
import numpy as np
import torchaudio
import gc  # garbage collector

# Configure CUDA memory allocator before torch import
# Critical for performance and memory management with GPU
def _ensure_alloc_conf(default: str):
    """Checks for existing PYTORCH_CUDA_ALLOC_CONF, sets default if not found."""
    env_var_name = "PYTORCH_CUDA_ALLOC_CONF"
    existing_conf = os.getenv(env_var_name)
    if existing_conf:
        cleaned_conf = existing_conf.strip()
        if cleaned_conf == 'backend:cudaMallocAsync':
            console.print(f"[warning]WARNING:[/warning] Detected problematic {env_var_name}='{cleaned_conf}'. Overriding with default: '{default}'", style="warning")
            os.environ[env_var_name] = default # Override problematic value
        else:
            # Use print here as logging might not be configured yet
            console.print(f"[info]INFO:[/info] Using existing {env_var_name} from environment: '{cleaned_conf}'", style="info")
            # Ensure the environment variable remains as is (already set)
    else:
        console.print(f"[info]INFO:[/info] {env_var_name} not set in environment. Setting default: '{default}'", style="info")
        os.environ[env_var_name] = default # Set it only if not present

# Ensure the allocator config is set (or confirmed) *before* importing torch
_ensure_alloc_conf("max_split_size_mb:128")

# Now it's safe to import torch
import torch

# ASR model and processor management
# Handles hardware detection, model loading and memory management
# Prioritizes CUDA > MPS > CPU with appropriate data types
def detect_device_and_dtype():
    """Detects hardware, determines effective device based on config, and selects appropriate torch dtype considering overrides."""
    # 1. Detect actual hardware availability
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    has_rocm = getattr(torch.version, "hip", None) is not None
    gpu_name = "None"
    detected_backend = "CPU"
    detected_device = "cpu"

    if has_cuda:
        detected_device = "cuda:0"
        detected_backend = "CUDA"
        gpu_name = torch.cuda.get_device_name(0)
    elif has_mps:
        detected_device = "mps"
        detected_backend = "Apple Metal (MPS)"
        gpu_name = "Apple Silicon"
    elif has_rocm:
        detected_device = "cuda:0" # ROCm often aliases to cuda
        detected_backend = "ROCm (HIP)"
        gpu_name = "AMD GPU" # Placeholder

    logging.info(f"Detected Hardware Backend: {detected_backend}")
    logging.info(f"Detected Hardware Device: {detected_device}")
    if gpu_name != "None":
        logging.info(f"Detected Hardware GPU Name: {gpu_name}")

    # 2. Determine effective device based on DEVICE config setting stored in Config object
    device_setting = Config.DEVICE_SETTING # Use the attribute from config.py
    effective_device = "cpu"
    effective_backend = "CPU"
    effective_gpu_name = "None"

    if device_setting == 'cpu':
        logging.info("DEVICE config set to 'cpu'. Forcing CPU usage.")
        effective_device = "cpu"
        effective_backend = "CPU"
    elif device_setting == 'gpu':
        if has_cuda or has_mps or has_rocm:
            effective_device = detected_device
            effective_backend = detected_backend
            effective_gpu_name = gpu_name
            logging.info(f"DEVICE config set to 'gpu'. Using detected GPU: {effective_backend} ({effective_device})")
        else:
            logging.warning("DEVICE config set to 'gpu', but no compatible GPU detected. Falling back to CPU.")
            effective_device = "cpu"
            effective_backend = "CPU"
    elif device_setting == 'auto':
        effective_device = detected_device
        effective_backend = detected_backend
        effective_gpu_name = gpu_name
        logging.info(f"DEVICE config set to 'auto'. Using best available device: {effective_backend} ({effective_device})")
    else:
        logging.warning(f"Invalid DEVICE setting '{device_setting}'. Defaulting to 'auto'.")
        effective_device = detected_device # Fallback to auto logic
        effective_backend = detected_backend
        effective_gpu_name = gpu_name

    # 3. Determine default dtype based on the *effective* device
    if effective_device == 'cpu':
        # CPU strongly prefers float32
        default_dtype = torch.float32
    elif effective_device.startswith("cuda") or effective_device == "mps":
        # GPUs generally benefit from float16 for memory, default to it
        default_dtype = torch.float16
    else: # Should not happen with current logic, but fallback
        default_dtype = torch.float32
    logging.info(f"Default dtype for effective device '{effective_device}': {default_dtype}")

    # 4. Check for dtype override and validate compatibility *against effective device*
    dtype_override_str = Config.TORCH_DTYPE_OVERRIDE
    final_dtype = default_dtype # Start with the default

    if dtype_override_str != 'auto':
        logging.info(f"Found TORCH_DTYPE_OVERRIDE='{dtype_override_str}' in config.")
        requested_dtype = None
        is_compatible = True
        can_override_cpu = False # Flag to check if override is allowed on CPU

        if dtype_override_str == 'float16':
            requested_dtype = torch.float16
            # Allow on GPU/MPS, but generally not beneficial on CPU
            if effective_device != 'cpu':
                 is_compatible = True
            else:
                 is_compatible = False # Disallow float16 override on CPU by default
                 logging.warning(f"TORCH_DTYPE_OVERRIDE='float16' is ignored when DEVICE='cpu'. Using default CPU dtype ({default_dtype}).")
        elif dtype_override_str == 'float32':
            requested_dtype = torch.float32
            is_compatible = True # Universally compatible
            can_override_cpu = True # Explicitly allow float32 override on CPU (though it's the default)
        elif dtype_override_str == 'bfloat16':
            requested_dtype = torch.bfloat16
            # Check compatibility with effective device
            if effective_device.startswith("cuda"):
                if not torch.cuda.is_bf16_supported():
                    is_compatible = False
                    logging.warning(f"bfloat16 override requested, but not supported by current CUDA device ('{effective_gpu_name}').")
            elif effective_device == "mps":
                 is_compatible = False
                 logging.warning("bfloat16 override requested, but generally not well-supported on MPS.")
            elif effective_device == "cpu":
                 # bfloat16 on CPU is possible but often not optimal for Whisper-like models. Disallow by default.
                 is_compatible = False
                 logging.warning(f"TORCH_DTYPE_OVERRIDE='bfloat16' is ignored when DEVICE='cpu'. Using default CPU dtype ({default_dtype}).")

        # Apply the override only if valid, compatible, and allowed for the device
        if requested_dtype is None:
             logging.warning(f"Invalid TORCH_DTYPE_OVERRIDE value: '{dtype_override_str}'. Ignoring override.")
        elif not is_compatible:
             logging.warning(f"Override '{dtype_override_str}' is not compatible or ignored for effective device '{effective_device}'. Falling back to default dtype '{default_dtype}'.")
             # final_dtype remains default_dtype
        else:
             # Check if it's a crazy CPU override attempt for non-float32
             if effective_device == 'cpu' and not can_override_cpu:
                  # This case should be caught by is_compatible=False above, but double-check
                  logging.warning(f"Ignoring incompatible override '{dtype_override_str}' for CPU. Using {default_dtype}.")
             elif requested_dtype != default_dtype:
                 final_dtype = requested_dtype
                 logging.info(f"Applying TORCH_DTYPE_OVERRIDE '{dtype_override_str}' (final dtype: {final_dtype}).")
             else:
                 logging.info(f"TORCH_DTYPE_OVERRIDE '{dtype_override_str}' matches default dtype for effective device. No change needed.")

    else:
        logging.info("TORCH_DTYPE_OVERRIDE is 'auto'. Using default dtype for effective device.")

    # Log final to let people know the selected dtype and effective device
    logging.info(f"--- Effective Device for Transcription: {effective_device} ---")
    logging.info(f"--- Final torch dtype selected: {final_dtype} ---")

    # Return effective device, final dtype, effective backend name, and effective GPU name
    return effective_device, final_dtype, effective_backend, effective_gpu_name

# Detect hardware capabilities globally
device, torch_dtype, backend, gpu_name = detect_device_and_dtype()

_GLOBAL_PIPELINE_INSTANCE = None
_PIPELINE_LOCK = multiprocessing.Lock()  # Lock for safe access/modification

# Singleton ASR model and processor with thread-safe initialization
# Cached globally to avoid repeated model loading
_MODEL_AND_PROCESSOR = None
_MODEL_LOCK = multiprocessing.Lock()

def get_model_and_processor():
    """Initialize and cache the ASR model and processor with enhanced loading."""
    global _MODEL_AND_PROCESSOR
    with _MODEL_LOCK:
        if _MODEL_AND_PROCESSOR is not None:
            return _MODEL_AND_PROCESSOR

        logging.info("Initializing ASR model and processor with Accelerate...")
        try:
            from huggingface_hub import snapshot_download
            checkpoint = Config.TRANSCRIBE_MODEL_NAME
            # Disable tqdm progress bar for snapshot_download using documented argument
            logging.debug("Calling snapshot_download with tqdm_class=None.")
            weights_location = snapshot_download(repo_id=checkpoint, tqdm_class=None)

            # Use the globally determined dtype and device
            model_dtype = torch_dtype
            logging.info(f"Attempting to load model with dtype: {model_dtype} onto device: {device}")

            load_args = {
                "torch_dtype": model_dtype,
                "use_safetensors": True,
            }

            if device != "cpu":
                # For GPU/MPS, let from_pretrained handle accelerate integration
                load_args["low_cpu_mem_usage"] = True
                # Use the effective device string ('cuda:0', 'mps') for device_map
                load_args["device_map"] = device
                logging.info(f"Using from_pretrained with low_cpu_mem_usage=True and device_map='{device}'")
            else:
                # For CPU, load directly without device_map (should default to CPU)
                # low_cpu_mem_usage=False is the default and appropriate here.
                # Explicitly set low_cpu_mem_usage to False for clarity if needed, but default is False.
                # load_args["low_cpu_mem_usage"] = False # Optional: Explicitly set
                logging.info("Using from_pretrained for direct CPU loading (low_cpu_mem_usage=False).")

            # Load the model using from_pretrained with the configured arguments
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                checkpoint, # Checkpoint name or path from config
                **load_args
            )

            # For CPU, double-check it loaded correctly (usually does without device_map)
            # For GPU/MPS, device_map should handle placement.
            if device == "cpu" and str(model.device) != 'cpu':
                 logging.warning(f"Model loaded to {model.device} despite CPU target. Forcing model.to('cpu').")
                 model = model.to(device) # Ensure it's on CPU

            logging.info(f"Model successfully loaded onto device: {model.device}")

            # Gradient checkpointing (Optional)
            # NB: This might require model = accelerator.prepare(model) earlier if using full Accelerate features.
            if device != "cpu" and Config.accelerator.is_main_process: # Check if distributed setup exists
                 try:
                     model.gradient_checkpointing_enable()
                     logging.info("Enabled gradient checkpointing for potential distributed use.")
                 except AttributeError:
                     logging.warning("Model does not support gradient_checkpointing_enable(). Skipping.")


            processor = AutoProcessor.from_pretrained(Config.TRANSCRIBE_MODEL_NAME)
            _MODEL_AND_PROCESSOR = (model, processor)

            logging.info(f"ASR model and processor initialized successfully.")
            return _MODEL_AND_PROCESSOR

        except Exception as e:
            logging.error(f"Failed to load ASR model with Accelerate: {e}", exc_info=True)
            console.print(f"[error][FATAL][/error] Could not load ASR model with Accelerate: {e}", style="error")
            return None, None

# Clean up ASR model resources and free GPU memory
# Critical for long-running processes to prevent memory leaks and make VRAM available for Ollama or vLLM
def release_model_and_processor():
    """Safely releases ASR model and processor from memory and clears GPU cache."""
    global _MODEL_AND_PROCESSOR
    with _MODEL_LOCK:
        if _MODEL_AND_PROCESSOR is not None:
            logging.info("Releasing ASR model and processor...")
            try:
                model, _ = _MODEL_AND_PROCESSOR
                if model is not None:
                    del model
                del _MODEL_AND_PROCESSOR
                _MODEL_AND_PROCESSOR = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logging.info("Model and processor released successfully.")
            except Exception as e:
                logging.error(f"Error during model release: {e}", exc_info=True)
        else:
            logging.info("No ASR model currently loaded.")

# Audio processing pipeline
# Handles sample rate conversion and mono conversion

def load_and_preprocess_audio(file_path: str, target_sr=Audio_Config.TARGET_SAMPLE_RATE):
    """Standardizes audio format for ASR pipeline - mono channel, target sample rate"""
    logging.debug(f"Loading audio file: {file_path}")
    try:
        waveform, sr = torchaudio.load(file_path)
        logging.debug(f"Original shape: {waveform.shape}, Original SR: {sr}")
        # If multi-channel, average to mono.
        if waveform.shape[0] > 1:
            logging.debug("Audio is multi-channel, averaging to mono.")
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if necessary
        if sr != target_sr:
            logging.debug(f"Resampling from {sr} Hz to {target_sr} Hz")
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
            sr = target_sr
        # Squeeze to 1D numpy array
        logging.debug(f"Final shape: {waveform.shape}")
        return waveform.squeeze(0).numpy(), sr
    except Exception as e:
        logging.error(f"Error loading/processing audio file {file_path}: {e}", exc_info=True)
        console.print(f"[error]Skipping file {os.path.basename(file_path)} due to audio loading error.[/error]", style="error")
        return None, None

# Split long audio files into manageable chunks
# Uses silence detection with fallback to fixed chunking
def split_audio_file(file_path: str, max_duration_s: int):
    """Splits long audio into chunks using silence detection with fallback to fixed chunks
    to prevent OOM errors during transcription"""
    logging.info(f"Attempting to split audio file: {file_path}")
    y, sr = load_and_preprocess_audio(file_path)
    if y is None or sr is None:
        return []
    segments = []
    try:
        # Split by silence first (adjust top_db as needed)
        logging.debug(f"Splitting by silence (top_db=30)")
        intervals = librosa.effects.split(y, top_db=30)
        max_samples = int(max_duration_s * sr)  # Use integer samples
        logging.debug(f"Max samples per chunk: {max_samples} ({max_duration_s}s at {sr}Hz)")

        for start_i, end_i in intervals:
            segment_y = y[start_i:end_i]
            # If a silence-based segment is too long, chunk it further
            if len(segment_y) > max_samples:
                logging.debug(f"Silence-based segment ({start_i}-{end_i}) is longer than max duration ({len(segment_y)} > {max_samples}), chunking further.")
                for i in range(0, len(segment_y), max_samples):
                    chunk = segment_y[i: i + max_samples]
                    if len(chunk) > 0:  # Avoid empty chunks
                        segments.append(chunk)
                        logging.debug(f"  Added fixed chunk of size {len(chunk)}")
            elif len(segment_y) > 0:  # Add non-empty short segments
                segments.append(segment_y)
                logging.debug(f"Added silence-based segment of size {len(segment_y)}")

        # Convert numpy segments to torch tensors (CPU first)
        # Ensure segments are float32, unsqueeze adds channel dim back temporarily
        torch_segments = [torch.tensor(seg, dtype=torch.float32).unsqueeze(0) for seg in segments]
        logging.info(f"Split {os.path.basename(file_path)} into {len(torch_segments)} segments.")
        return torch_segments
    except Exception as e:
        logging.error(f"Error splitting audio file {file_path}: {e}", exc_info=True)
        console.print(f"[error]Error splitting audio file {os.path.basename(file_path)}. Skipping.[/error]", style="error")
        return []

# Core transcription function with error handling and retries
# Dynamically adjusts batch size and chunk length on OOM errors

def transcribe_from_array(audio_array, current_batch_size, file_path: str = "Unknown File") -> (str, int):
    """Core transcription with adaptive retries - reduces batch size on OOM errors
    to maximize success rate while maintaining performance"""
    transcription = ""

    if not isinstance(audio_array, np.ndarray):
        logging.error(f"Invalid input type for transcription: {type(audio_array)}. Expected numpy array.")
        return "INVALID_INPUT", current_batch_size

    if audio_array.ndim > 1:
        audio_array = np.squeeze(audio_array)
    if len(audio_array) == 0:
        logging.warning("Attempted to transcribe empty audio array.")
        return "", current_batch_size # Return only batch size

    model, processor = get_model_and_processor()
    if model is None or processor is None:
        logging.error("ASR model/processor unavailable.")
        return "PIPELINE_INIT_FAILED", current_batch_size

    local_batch_size = current_batch_size
    audio_config = AudioConfig()
    generate_kwargs = {
        "max_length": audio_config.TRANSCRIBE_MAX_LENGTH,
        "num_beams": audio_config.TRANSCRIBE_NUM_BEAMS,
    }

    if Config.TRANSCRIBE_LANGUAGE:
        try:
            if processor:
                forced_decoder_ids = processor.get_decoder_prompt_ids(language=Config.TRANSCRIBE_LANGUAGE, task="transcribe")
                generate_kwargs["forced_decoder_ids"] = forced_decoder_ids
                logging.debug(f"Forcing language to '{Config.TRANSCRIBE_LANGUAGE}' using decoder IDs.")
            else:
                logging.warning("Processor not available, cannot force language.")
        except Exception as lang_err:
            logging.warning(f"Could not get forced_decoder_ids for language '{Config.TRANSCRIBE_LANGUAGE}': {lang_err}. Proceeding without forcing.")


    for attempt in range(Config.RETRY_TRANSCRIBE + 1):
        try:
            # Ensure processor is available before using it
            if not processor:
                 logging.error("Processor became unavailable during retry loop.")
                 return "PIPELINE_PROCESSOR_LOST", current_batch_size

            inputs = processor(
                audio_array,
                sampling_rate=Audio_Config.TARGET_SAMPLE_RATE,
                return_tensors="pt",
                return_attention_mask=True
                # language=Config.TRANSCRIBE_LANGUAGE # Removed: Not needed here, handled by forced_decoder_ids
            )
            # Send tensors explicitly to the globally determined effective device
            input_features = inputs["input_features"].to(dtype=model.dtype, device=device)
            attention_mask = inputs["attention_mask"].to(device=device)

            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Input features should already be on the correct device and dtype from the processor
                # and model loading logic.
                predicted_ids = model.generate(
                    input_features, # Use features directly
                    attention_mask=attention_mask,
                    **generate_kwargs
                )
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

            log_msg = f"Transcription successful for chunk in '{os.path.basename(file_path)}' (Attempt {attempt + 1})"
            logging.debug(log_msg)
            return transcription, local_batch_size # Return only batch size

        except torch.cuda.OutOfMemoryError as oom_error:
            logging.error(f"OOM Error on attempt {attempt + 1} for chunk in '{os.path.basename(file_path)}': {oom_error}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if attempt < Config.RETRY_TRANSCRIBE:
                local_batch_size = max(1, local_batch_size // 2)
                logging.warning(f"OOM Retry {attempt + 1}: Reducing batch size to {local_batch_size}.")
                # No dtype casting needed here anymore
                continue
            else:
                logging.error(f"OOM Error final attempt failed for chunk in '{os.path.basename(file_path)}'.")
                return "OOM_FINAL", current_batch_size # Return only batch size

        except Exception as e:
            logging.error(f"Unexpected error during transcription attempt {attempt + 1} for chunk in '{os.path.basename(file_path)}': {e}", exc_info=True)
            # No retry flag to reset
            if attempt == Config.RETRY_TRANSCRIBE:
                logging.error(f"Generic error final attempt failed for chunk in '{os.path.basename(file_path)}'.")
                return "GENERIC_ERROR", current_batch_size # Return only batch size

    logging.error(f"Unknown error after all retries for chunk in '{os.path.basename(file_path)}'.")
    return "UNKNOWN_ERROR", current_batch_size # Return only batch size

def transcribe_audio_file(file_path: str):
    """Simplified transcription path for short files - bypasses chunking overhead
    when audio splitting is disabled"""
    logging.info(f"Transcribing {os.path.basename(file_path)} as a single chunk...")
    audio_array, sr = load_and_preprocess_audio(file_path)
    if audio_array is None:
        return ""  # Error logged in load_and_preprocess_audio

    # Use initial config values
    initial_batch_size = Config.AUDIO_BATCH_SIZE
    # initial_chunk_length = Config.CHUNK_LENGTH_S # Removed

    # Pass file_path to transcribe_from_array for logging
    transcription, _ = transcribe_from_array(audio_array, initial_batch_size, file_path=file_path)

    if "ERROR" in transcription or "FAILED" in transcription or "INVALID" in transcription:
        logging.error(f"Transcription failed for single chunk: {file_path}. Reason: {transcription}")
        return ""
    return transcription

# Main transcription workflow with optional audio splitting
# Designed to handle both short and long audio files
def transcribe_large_audio(file_path: str, master_progress): # Keep master_progress object
    """Main transcription workflow - handles both chunked and unchunked processing
    while maintaining clean separation from UI concerns. Updates a temporary chunk progress bar."""
    # If splitting is explicitly set to 'no', transcribe the whole file. Otherwise, split.
    if Config.SPLIT_AUDIO == 'no':
        logging.info("Audio splitting explicitly disabled (SPLIT_AUDIO=no), transcribing file as a whole.")
        return transcribe_audio_file(file_path)
    else:
        # Proceed with splitting if SPLIT_AUDIO is 'yes' or any other value (defaulting to splitting)
        logging.info(f"Audio splitting enabled (SPLIT_AUDIO={Config.SPLIT_AUDIO}). Splitting file...")
        segments = split_audio_file(file_path, Config.MAX_CHUNK_DURATION_S)
    if not segments:
        logging.error(f"Audio splitting yielded no segments for {file_path}.")
        return ""  # Error logged in split_audio_file

    # Initialize list for this specific file's transcript parts
    full_transcript_parts = []
    current_batch_size = Config.AUDIO_BATCH_SIZE
    # current_chunk_length = Config.CHUNK_LENGTH_S # Removed
    any_chunk_failed = False
    # Removed .clear() as we are initializing fresh here

    logging.info(f"Transcribing {os.path.basename(file_path)} in {len(segments)} chunks...")
    model, processor = get_model_and_processor()
    if model is None or processor is None:
        logging.critical("ASR model or processor failed to initialize before chunk processing loop.")
        return "PIPELINE_INIT_FAILED"

    # Add a task for chunks of this specific file (make it non-transient to ensure visibility)
    chunk_task_id = master_progress.add_task(f"  Chunks: {os.path.basename(file_path)}", total=len(segments), transient=False)

    # Ensure the list holding transcript parts is empty before processing chunks for this file
    full_transcript_parts = []

    try: # Use try/finally to ensure task removal
        # Process chunks and update the temporary chunk progress bar
        for i, segment_tensor in enumerate(segments):
            logging.debug(f"Processing chunk {i+1}/{len(segments)}") # Chunk progress logged only in debug
            audio_array = np.squeeze(segment_tensor.numpy())
            # Add logging for chunk duration (indented correctly inside the loop)
            chunk_duration_s = len(audio_array) / Audio_Config.TARGET_SAMPLE_RATE
            logging.debug(f"Processing chunk {i+1}/{len(segments)} - Duration: {chunk_duration_s:.2f}s")
            chunk_transcript, new_batch_size = transcribe_from_array(
                audio_array, current_batch_size, file_path=file_path
            )

            # Update chunk progress bar regardless of chunk success/failure
            master_progress.update(chunk_task_id, advance=1)

            if new_batch_size < current_batch_size:
                logging.info(f"Reducing batch size for subsequent chunks to {new_batch_size}")
                current_batch_size = new_batch_size
        # if new_chunk_length < current_chunk_length: # Removed
        #     logging.info(f"Reducing chunk length for subsequent chunks to {new_chunk_length}s") # Removed
        #     current_chunk_length = new_chunk_length # Removed

            # Check if the chunk transcription failed or resulted in empty text (INDENTED CORRECTLY)
            if not chunk_transcript or "ERROR" in chunk_transcript or "FAILED" in chunk_transcript or "INVALID" in chunk_transcript:
                # Log appropriately based on whether it's an error string or just empty
                if not chunk_transcript:
                    logging.warning(f"Transcription for chunk {i+1}/{len(segments)} resulted in empty text for {file_path}.")
                else:
                    logging.error(f"Failed to transcribe chunk {i+1}/{len(segments)} for {file_path}. Reason: {chunk_transcript}")
                logging.debug(f'>>> Skipped appending chunk {i+1} due to error/empty.') # Log skipping
                any_chunk_failed = True
                # Do not append failed or empty chunks
            else:
                # This block executes only if chunk_transcript is valid and non-empty
                full_transcript_parts.append(chunk_transcript)
                logging.debug(f"Chunk {i+1} transcript length: {len(chunk_transcript)}")
                logging.debug(f'>>> Appended chunk {i+1}. Current parts count: {len(full_transcript_parts)}. Snippet: {chunk_transcript[:50]}...') # Log appending
        # End of the for loop (correct placement)
    finally:
        # This block always executes, ensuring the task is removed
        if 'chunk_task_id' in locals() and chunk_task_id is not None: # Check if task exists before removing
             try:
                 master_progress.remove_task(chunk_task_id)
             except Exception as e: # Catch potential errors if task is already gone
                 logging.debug(f"Could not remove chunk task {chunk_task_id}: {e}")


    if any_chunk_failed:
        warning_msg = f"Some chunks failed during transcription for {os.path.basename(file_path)}. The output may be incomplete."
        logging.warning(warning_msg)

    return " ".join(full_transcript_parts).strip()

def read_prompt_template(file_path: str) -> str:
    """Read and return the contents of a prompt template file.

    Args:
        file_path: Path to the prompt template file to read.

    Returns:
        The file contents as a string, or empty string on failure.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        UnicodeDecodeError: If file contains invalid UTF-8 characters.

    NB:
        Errors are logged and printed to stderr before returning empty string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logging.error(f"Prompt template file not found: {file_path}")
        console.print(f"\n[error]Prompt template file not found: {file_path}[/error]", style="error")
        return ""
    except Exception as e:
        logging.error(f"Error reading prompt template {file_path}: {e}", exc_info=True)
        console.print(f"\n[error]Could not read prompt template file: {e}[/error]", style="error")
        return ""

def refine_transcript_text(llm_api: LLMApiClient, transcript_text: str, prompt_template: str, progress = None):
    """Refine raw transcript using LLM with context-preserving chunk processing.

    Processes text in overlapping chunks to maintain contextual continuity.
    Implements deduplication of overlapping regions in refined output.

    Args:
        llm_api (LLMApiClient): Initialized LLM API client instance.
        transcript_text: Raw transcript text to refine
        prompt_template: Template string with {chunk}, {part_number}, {total_parts}
        progress: Rich progress tracker (optional)

    Returns:
        str: Combined and deduplicated refined text, or original text if refinement fails.

    NB:
        Uses Config.FRAGMENT_SIZE and Config.CHUNK_OVERLAP for chunk sizing
        Preserves original text in case of LLM failures
    """
    if not transcript_text or not prompt_template:
        logging.warning("Cannot refine empty transcript or without prompt template.")
        return transcript_text # Return original if input invalid

    # NEW: Overlapping chunks
    fragments = chunk_with_overlap(
        transcript_text,
        Config.FRAGMENT_SIZE,
        Config.CHUNK_OVERLAP
    )

    refined_fragments = []
    llm_failed_fragments = 0
    logging.info(f"Refining transcript ({len(transcript_text)} chars) in {len(fragments)} overlapping fragments...")

    task_id = progress.add_task("[cyan]Refining fragments...", total=len(fragments), visible=True, phase="Refining") if progress else None

    for i, fragment in enumerate(fragments):
        if progress:
            progress.update(task_id, advance=1, description=f"[cyan]Refining fragment {i+1}/{len(fragments)}")
        prompt = prompt_template.format(
            part_number=i + 1,
            total_parts=len(fragments),
            chunk=fragment
        )
        logging.debug(f"Processing fragment {i+1}/{len(fragments)}")
        refined_fragment = llm_api.process_text(prompt)

        # Check for None return value from process_text (indicates failure or empty response)
        if refined_fragment is None:
            logging.warning(f"LLM refinement failed for fragment {i + 1}. Using original fragment.")
            refined_fragments.append(fragment) # Use original fragment on failure
            llm_failed_fragments += 1
        else:
            refined_fragments.append(refined_fragment)

    if llm_failed_fragments > 0:
        warning_msg = f"{llm_failed_fragments}/{len(fragments)} fragments failed refinement. Original text used for failed fragments."
        logging.warning(warning_msg)
        # Optionally print to console if needed: console.print(f"[warning]{warning_msg}[/]")

    # Deduplicate overlapping content even if some fragments failed
    if not refined_fragments: # Handle case where all fragments failed
         logging.error("All fragments failed refinement. Returning original text.")
         return transcript_text # Or return "" depending on desired behavior

    deduped_fragments = [refined_fragments[0]]
    for i in range(1, len(refined_fragments)):
        deduped = trim_redundant_overlap(deduped_fragments[-1], refined_fragments[i])
        deduped_fragments.append(deduped)

    final_text = "\n\n".join(deduped_fragments)
    # Return original text if the final joined text is empty after processing
    return final_text if final_text else transcript_text


def chunk_with_overlap(text: str, size: int, overlap: int) -> list[str]:
    """Split text into chunks with specified overlap between them.

    Args:
        text: The input text to split
        size: Maximum size of each chunk (in characters)
        overlap: Number of characters to overlap between chunks

    Returns:
        A list of text chunks with the specified overlap

    Raises:
        ValueError: If size <= 0 or overlap is invalid

    Example:
        >>> chunk_with_overlap("abcdefghij", 4, 2)
        ['abcd', 'cdef', 'efgh', 'ghij']
    """
    if size <= 0:
        raise ValueError("Chunk size must be greater than 0")
    if overlap < 0:
        raise ValueError("Overlap must be non-negative")
    if overlap >= size:
        raise ValueError("Overlap must be smaller than chunk size")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)

        if end >= len(text):
            break
        start = end - overlap  # Move start back by overlap

    return chunks

def trim_redundant_overlap(a: str, b: str, min_overlap: int = 20) -> str:
    """Remove redundant overlapping text between two consecutive chunks.

    Args:
        a: Previous chunk text
        b: Current chunk text
        min_overlap: Minimum overlap to check for (default: 20)

    Returns:
        Deduplicated version of current chunk text
    """
    max_check = min(len(a), len(b), 200)  # Limit scan window
    for i in range(max_check, min_overlap - 1, -1):
        if a[-i:].strip() == b[:i].strip():
            return b[i:].lstrip()
    return b

# Core file processing workflow:
# 1. Validate input
# 2. Transcribe audio
# 3. Optionally refine with LLM
# 4. Save output

# Phase 1 processing - handles raw transcription
# Kept separate from refinement for modularity
def process_single_file_transcription(input_file: str, output_file: str, master_progress) -> bool:
    """ Phase 1: Transcribes a single audio file and saves the RAW transcript. """
    # print("-" * 60) # Keep console clean
    logging.info(f"Starting PHASE 1 (Transcription) for: {os.path.basename(input_file)}")
    # print(f"Transcribing: {os.path.basename(input_file)}") # Keep console clean

    if not os.path.exists(input_file):
        logging.error(f"Input audio file not found: {input_file}")
        console.print(f"[error]File not found: {input_file}. Skipping.[/error]", style="error")
        return False  # Indicate failure

    # Transcription Step
    # Pass only master_progress
    full_transcript = transcribe_large_audio(input_file, master_progress)

    if "PIPELINE_INIT_FAILED" in full_transcript:
        console.print("[error][FATAL] ASR pipeline failed to initialize. Cannot continue transcription phase.[/error]", style="error")
        # Signal to stop the whole process might be needed here depending on desired behavior
        return False  # Indicate fatal error
    elif not full_transcript or "ERROR" in full_transcript or "FAILED" in full_transcript or "INVALID" in full_transcript:
        logging.error(f"Transcription resulted in empty or error output for {input_file}. Reason: {full_transcript}")
        console.print(f"[error]Transcription failed or produced empty/error output for {os.path.basename(input_file)}. Skipping file.[/error]", style="error")
        return False  # Indicate failure for this file

    logging.info(f"Raw transcription length: {len(full_transcript)}")
    logging.debug(f"Raw transcript snippet: {full_transcript[:200]}...")

    # Save RAW Output Step
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir:  # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            logging.debug(f"Ensured output directory exists: {output_dir}")

        with open(output_file, 'w', encoding='utf-8') as md_file:
            md_file.write(full_transcript)
        logging.info(f"Successfully saved RAW transcript to {output_file}")
        # print(f"Successfully saved RAW transcript: {os.path.basename(output_file)}") # Keep console clean
        return True  # Indicate success for Phase 1
    except Exception as e:
        logging.error(f"Error saving RAW transcript to {output_file}: {e}", exc_info=True)
        console.print(f"[error]Could not save RAW transcript to {output_file}: {e}[/error]", style="error")
        return False  # Indicate failure

def process_single_file_refinement(llm_api: LLMApiClient, input_transcript_path: str, target_output_path: str, prompt_template: str) -> bool:
    """Apply LLM refinement to a transcript file (Phase 2 processing).

    Reads the transcript from input_transcript_path, processes it with the LLM,
    and saves the result to target_output_path.

    Args:
        llm_api (LLMApiClient): Initialized LLM API client instance.
        input_transcript_path (str): Path to the raw transcript file to read.
        target_output_path (str): Path where the refined transcript should be saved.
        prompt_template (str): Template string for LLM refinement.

    Returns:
        bool: True if refinement succeeded and was saved, False otherwise.

    Raises:
        FileNotFoundError: If input transcript file is missing.
        IOError: For file access issues during read/write.
    """
    logging.info(f"Starting PHASE 2 (Refinement) for: {os.path.basename(input_transcript_path)}")
    logging.info(f"  Output will be saved to: {os.path.basename(target_output_path)}")
    # Use target_output_path for user feedback
    print(f"Refining: {os.path.basename(input_transcript_path)} -> {os.path.basename(target_output_path)}")

    # LLM Client should already be initialized and passed in
    if not llm_api.is_available():
        logging.error("LLM Client not available for refinement phase.")
        console.print(f"  [error]LLM Client not available. Skipping refinement for {os.path.basename(input_transcript_path)}.[/error]", style="error")
        return False

    try:
        # Read the raw transcript from the input path
        with open(input_transcript_path, 'r', encoding='utf-8') as f:
            raw_transcript_text = f.read()
        logging.debug(f"Read {len(raw_transcript_text)} chars from {input_transcript_path} for refinement.")

        if not raw_transcript_text:
            logging.warning(f"Input transcript file is empty: {input_transcript_path}. Skipping refinement.")
            console.print(f"  [warning]Input transcript file is empty. Skipping refinement for {os.path.basename(input_transcript_path)}.[/warning]", style="warning")
            return False # Consider this a skip

        # Refine the text using the passed LLM client instance
        refined_transcript = refine_transcript_text(llm_api, raw_transcript_text, prompt_template, progress=None)

        # Check for None return value or if it reverted to original (indicating failure)
        if refined_transcript is None or refined_transcript == raw_transcript_text:
            if refined_transcript is None:
                 logging.error(f"LLM refinement failed completely for {input_transcript_path}. No output generated.")
                 console.print(f"  [error]LLM refinement failed completely for {os.path.basename(input_transcript_path)}. No output generated.[/error]", style="error")
            else:
                 # This means refine_transcript_text returned the original due to partial failures
                 logging.warning(f"LLM refinement failed for one or more chunks in {input_transcript_path}. Saving partially refined/original text.")
                 console.print(f"  [warning]LLM refinement failed for one or more chunks in {os.path.basename(input_transcript_path)}. Saving partially refined/original text to {os.path.basename(target_output_path)}.[/warning]", style="warning")
            # Proceed to save even if partially failed, unless it was a total failure (None)
            if refined_transcript is None:
                 return False # Indicate total failure

        # Save the refined transcript to the TARGET output path
        output_dir = os.path.dirname(target_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True) # Ensure directory exists

        with open(target_output_path, 'w', encoding='utf-8') as md_file:
            md_file.write(refined_transcript) # Write refined (or partially refined) text

        logging.info(f"Successfully saved REFINED transcript to {target_output_path}")
        print(f"Successfully saved REFINED transcript: {os.path.basename(target_output_path)}") # Keep user feedback
        return True

    except FileNotFoundError:
        logging.error(f"Input transcript file not found for refinement: {input_transcript_path}")
        console.print(f"[error]Input transcript file not found for refinement: {os.path.basename(input_transcript_path)}. Skipping.[/error]", style="error")
        return False
    except Exception as e:
        # Use target_output_path in error message
        logging.error(f"Error during refinement or saving for {input_transcript_path} -> {target_output_path}: {e}", exc_info=True)
        console.print(f"[error]Could not refine or save transcript {os.path.basename(target_output_path)}: {e}[/error]", style="error")
        return False

# Main CLI workflow with:
# - Argument parsing
# - Input/output path handling
# - Progress UI with Rich
# - Summary reporting

    # Disable huggingface_hub progress bars to avoid conflict with Rich
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

def run():
    """Main entry point for the audio-to-markdown transcription with optional LLM refinement.

    Coordinates the complete processing workflow including:
    - Command-line argument parsing
    - Hardware detection and model initialization
    - File processing pipeline management
    - Progress tracking and reporting

    Handles both single file and batch processing modes with:
    - Automatic audio transcription
    - Optional LLM-based refinement
    - Comprehensive error handling
    """

    if len(sys.argv) not in [2, 3] or '--help' in sys.argv or '-h' in sys.argv:
        print("Usage:")
        print(f"  Single file: python {os.path.basename(sys.argv[0])} <audio_file.mp3> [<output_markdown.md>]")
        print(f"  Folder:      python {os.path.basename(sys.argv[0])} <input_folder> [<output_folder>]")
        sys.exit(1)

    input_path = sys.argv[1]
    # Determine if input is a directory *before* processing paths
    is_input_directory = os.path.isdir(input_path)
    output_path = None
    if len(sys.argv) == 3:
        output_path = sys.argv[2]
    logging.debug(f"Input path: {input_path}, Output path: {output_path}, Is directory: {is_input_directory}")

    # Detect hardware capabilities before logging setup and before using the variables
    # device, torch_dtype, backend, gpu_name = detect_device_and_dtype() # Already detected globally

    # Show Hardware Info to Console (moved after help check)
    print("-" * 60) # Separator
    print(f"INFO: Using backend: {backend}")
    if device != "cpu":
        print(f"INFO: Using device: {gpu_name} ({device})")
    else:
        print(f"INFO: Using device: CPU")
    print(f"INFO: Using data type: {torch_dtype}")
    print("-" * 60) # Separator

    # Log initial config values
    logging.info(f"ASR Model: {Config.TRANSCRIBE_MODEL_NAME}")
    logging.info(f"Transcript mode: {Config.TRANSCRIPT_MODE}")
    logging.info(f"Audio Splitting Enabled: {Config.SPLIT_AUDIO == 'yes'}")
    logging.info(f"Max Chunk Duration (s): {Config.MAX_CHUNK_DURATION_S}")
    logging.info(f"Initial Batch Size: {Config.AUDIO_BATCH_SIZE}")
    logging.info(f"Initial Chunk Length (s): {Config.CHUNK_LENGTH_S}")
    if Config.TRANSCRIPT_MODE == 'llm':
        # Log LLM config from TranscriptionConfig (used by LLMApiClient)
        logging.info(f"LLM Model: {Config.LLM_MODEL_NAME}")
        logging.info(f"LLM API Base URL: {Config.LLM_API_BASE_URL}")
        logging.info(f"LLM API Key Provided: {'Yes' if Config.LLM_API_KEY else 'No'}")
        logging.info(f"LLM Timeout: {Config.LLM_TIMEOUT}s")
        logging.info(f"LLM Retries: {Config.LLM_RETRIES}")
        logging.info(f"LLM Temperature: {Config.LLM_TEMPERATURE}")
        logging.info(f"LLM Fragment Size: {Config.FRAGMENT_SIZE}")
        logging.info(f"LLM Chunk Overlap: {Config.CHUNK_OVERLAP}")
    logging.info(f"Debug Logging Enabled: {Config.DEBUG}")
    logging.info(f"Using device: {device} with dtype: {torch_dtype}")
    # NB: PYTORCH_CUDA_ALLOC_CONF is set via environment variable before torch import

    # Define supported extensions for audio/video files
    supported_extensions = (".mp3", ".wav", ".flac", ".m4a", ".m4b", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mkv", ".avi", ".mov", ".wmv")

    # Determine input/output file paths using the refactored function
    # NB: determine_file_paths handles console output via the ui instance
    try:
        files_to_process_phase1, output_location_summary = determine_file_paths(
            input_path=input_path,
            output_suffix="_transcript", # Suffix for transcript files/folders
            allowed_extensions=supported_extensions,
            ui=ui, # Pass the initialized UI instance
            output_extension=".md" # Ensure output is always .md
        )
        # Output location summary is now handled within determine_file_paths via console.print
        # if output_location_summary:
        #     print(output_location_summary) # Or log it if preferred

    except SystemExit:
        # determine_file_paths calls sys.exit on errors, so we just exit here too
        sys.exit(1)
    except Exception as e:
        # Catch any unexpected errors during path determination
        logging.critical(f"An unexpected error occurred during file path determination: {e}", exc_info=True)
        ui.get_console().print(f"[error]An unexpected error occurred: {e}[/]", style="error")
        sys.exit(1)

    # The checks for empty list and printing the count are now inside determine_file_paths
    # if not files_to_process_phase1:
    #     # Logging and printing handled within determine_file_paths
    #     sys.exit(0)
    # print(f"\nFound {len(files_to_process_phase1)} audio file(s) to process.") # Handled by determine_file_paths

    # Store Stage 1 output folder if input was a directory
    stage1_output_folder = None
    if is_input_directory and files_to_process_phase1:
        # Infer Stage 1 output folder from the first processed file's output path
        stage1_output_folder = os.path.dirname(files_to_process_phase1[0][1])
        logging.info(f"Stage 1 output directory identified as: {stage1_output_folder}")

    successfully_transcribed_files = []  # Store paths of successfully created raw transcripts (Stage 1 outputs)
    transcription_failures = 0
    pipeline_failed = False

    # Configure logging using the centralized UI method
    script_base_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    # Create the main Live display components using the UI helper
    # This will create the actual log_panel instance we want logs to go to
    live, master_progress, log_panel = ui.create_live_display()
    # Configure logging to use the *actual* log_panel from the live display
    ui.setup_live_logging(log_panel, Config.DEBUG, script_base_name)

    # Further reduce verbosity from noisy libraries when not in debug mode
    if not Config.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        # Add other noisy libraries here if necessary



    # Detect hardware capabilities AFTER logging setup
    # device, torch_dtype, backend, gpu_name = detect_device_and_dtype() # Already detected globally

    # Set CUDA_LAUNCH_BLOCKING for debugging CUDA errors if enabled in config
    if Config.CUDA_LAUNCH_BLOCKING:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        logging.debug("Set CUDA_LAUNCH_BLOCKING=1 for debugging CUDA errors.")
    else:
        logging.debug("CUDA_LAUNCH_BLOCKING is disabled in config.")

    # Phase 1: Transcription
    print("\n--- Starting Phase 1: Transcription ---")
    logging.info(f"Starting Phase 1: Transcription for {len(files_to_process_phase1)} files.")

    # Start the Live display with slightly higher refresh rate
    with live:
        # Master task for overall job
        job_task = master_progress.add_task(
            "[bold cyan]Global Progress",
            total=len(files_to_process_phase1)*2, # Total steps: transcribe + refine (if applicable)
            phase="Overall"
        )

        # Transcription phase task (persistent)
        transcribe_task = master_progress.add_task(
            "[green]Transcribing Files...", # Initial generic description
            total=len(files_to_process_phase1),
            phase="Transcription"
        )

        for idx, (input_f, output_f) in enumerate(files_to_process_phase1):
            # Update overall progress description for the JOB task
            master_progress.update(job_task, description=f"[cyan]Overall: File {idx+1}/{len(files_to_process_phase1)}")
            # Update the Transcription task fields to show current file in progress bar
            master_progress.update(transcribe_task,
                                description=f"[green]Transcribing File {idx+1}/{len(files_to_process_phase1)}",
                                fields={"phase": f"{os.path.basename(input_f)}"})

            # Call processing function
            success = process_single_file_transcription(input_f, output_f, master_progress)

            # Advance the Transcription task after the file is processed
            master_progress.update(transcribe_task, advance=1)
            # Description will be updated for the next file in the next iteration

            master_progress.update(job_task, advance=1) # Advance overall job progress after transcription step for this file
            if success:
                successfully_transcribed_files.append(output_f)
            else:
                transcription_failures += 1
                # Check if the pipeline itself failed fatally
                # Re-check if model/processor is truly gone after a failure in transcription
                model_check, proc_check = get_model_and_processor() # Attempt to get again
                if model_check is None or proc_check is None:
                    # If it's still None after a failure, assume fatal pipeline issue
                    pipeline_failed = True
                    logging.critical("ASR Pipeline appears to have failed fatally. Aborting transcription phase.")
                    live.console.print("\n[bold red][FATAL][/bold red] ASR Pipeline initialization failed. Aborting transcription phase.")
                    break # Exit the transcription loop

        # Stop the transcription task explicitly if loop finished
        if not master_progress.tasks[transcribe_task].finished:
             master_progress.update(transcribe_task, description="[green]Transcription Phase Complete")
             master_progress.stop_task(transcribe_task)


        phase1_summary = f"Phase 1: Transcription Complete - Successfully transcribed: {len(successfully_transcribed_files)} file(s), Failed or skipped: {transcription_failures} file(s)"
        logging.info(phase1_summary)
        live.console.print("\n--- Phase 1: Transcription Complete ---") # Print below the live display
        live.console.print(f"  Successfully transcribed: {len(successfully_transcribed_files)} file(s)")
        live.console.print(f"  Failed or skipped:      {transcription_failures} file(s)")

    # Release ASR Pipeline
    # Use the main console, not the live one which has exited
    console.print("\nReleasing ASR model from memory...")
    release_model_and_processor()
    console.print("ASR model released.")

    # Phase 2: Refinement
    refinement_successes = 0
    refinement_failures = 0
    llm_api = None # Initialize LLM API client variable for this phase

    if Config.TRANSCRIPT_MODE == 'llm' and not pipeline_failed:
        if not successfully_transcribed_files:
            logging.info("No files were successfully transcribed. Skipping Phase 2 (Refinement).")
            console.print("\nNo files were successfully transcribed. Skipping Phase 2 (Refinement).")
        else:
            console.print("\n--- Starting Phase 2: LLM Refinement ---")
            logging.info("Starting Phase 2: LLM Refinement.")

            # Initialize LLM Client HERE, after ASR release
            logging.info("Initializing LLM API Client for refinement phase...")
            # Pass the instantiated config object
            llm_api = LLMApiClient(config=Config)

            if not llm_api.is_available():
                console.print("[error]LLM Client failed to initialize. Check config and logs.[/]", style="error")
                logging.critical("LLM Client initialization failed. Skipping refinement.")
                llm_api = None # Ensure it's None if init fails
            else:
                console.print("Attempting to warm up LLM...")
                if not llm_api.warm_up():
                    # Error should be logged by warm_up method
                    console.print("[error]LLM warm-up failed. Check connection/authentication in logs.[/]", style="error")
                    logging.critical("LLM warm-up failed. Skipping refinement.")
                    llm_api = None # Mark as unavailable if warm-up fails
                else:
                    console.print("[success]LLM service responded.[/]", style="success") # Print success message here

            if llm_api and llm_api.is_available(): # Proceed only if client initialized and warmed up
                # --- Prepare for Stage 2 ---
                stage2_output_folder = None
                timestamp_s2 = datetime.datetime.now().strftime("%Y%m%d_%H%M") # Timestamp for refinement stage

                # Create Stage 2 output folder if input was a directory
                if is_input_directory and stage1_output_folder:
                    stage2_output_folder = f"{stage1_output_folder}_orthography_{timestamp_s2}"
                    try:
                        os.makedirs(stage2_output_folder, exist_ok=True)
                        logging.info(f"Created Stage 2 output directory: {stage2_output_folder}")
                        console.print(f"[info]Saving refined output to:[/info] {stage2_output_folder}")
                    except Exception as e:
                        logging.error(f"Could not create Stage 2 output directory {stage2_output_folder}: {e}", exc_info=True)
                        console.print(f"[error]Could not create Stage 2 output directory {stage2_output_folder}. Skipping refinement.[/error]", style="error")
                        llm_api = None # Prevent refinement loop from running

                # Load prompt template (only if LLM API is still valid)
                if llm_api:
                    try:
                        prompt_path = Config.TRANSCRIPT_PROMPT
                        prompt_template = read_prompt_template(prompt_path)
                        if not prompt_template:
                            raise ValueError(f"Empty prompt template at '{prompt_path}'")
                    except Exception as e:
                        logging.error(f"Prompt template error: {str(e)}. Skipping Phase 2.")
                        console.print(f"\n[error]Prompt template error: {str(e)}. Skipping Phase 2.[/error]", style="error")
                        llm_api = None # Prevent refinement loop

                # --- Run Stage 2 Refinement Loop (if LLM API and prompt are valid) ---
                if llm_api:
                    logging.debug("Loaded prompt template for refinement.")
                    console.print(f"Refining {len(successfully_transcribed_files)} transcript(s)...")
                    # Re-enter Live context for refinement progress
                    with live: # Re-use the main live context
                        refine_task = master_progress.add_task(
                            "[magenta]Refining Files...", # Initial generic description
                            total=len(successfully_transcribed_files),
                            phase="Refinement",
                            visible=True # Make sure it's visible
                        )
                        # Loop through the successfully transcribed files (Stage 1 outputs)
                        for idx_r, transcript_f_s1 in enumerate(successfully_transcribed_files):
                             # Calculate the target output path for Stage 2
                             target_output_path_s2 = None
                             if is_input_directory and stage2_output_folder:
                                 # Directory Input: Construct path inside the Stage 2 folder
                                 # Use the *original* audio file's base name for the refined output filename
                                 # We need the original input path corresponding to this Stage 1 output path
                                 # The files_to_process_phase1 list stores (original_input, stage1_output) tuples
                                 original_input_audio_path = files_to_process_phase1[idx_r][0] # Get original audio path
                                 original_base_name, _ = os.path.splitext(os.path.basename(original_input_audio_path))
                                 target_filename_s2 = f"{original_base_name}_orthography_{timestamp_s2}.md"
                                 target_output_path_s2 = os.path.join(stage2_output_folder, target_filename_s2)
                             elif not is_input_directory:
                                 # Single File Input: Construct the double-suffixed filename
                                 base_name_s1, ext_s1 = os.path.splitext(transcript_f_s1) # transcript_f_s1 is Stage 1 output path
                                 target_output_path_s2 = f"{base_name_s1}_orthography_{timestamp_s2}{ext_s1}" # ext_s1 is .md
                             else:
                                 # Should not happen if directory creation succeeded, but safety check
                                 logging.error(f"Could not determine target Stage 2 path for {transcript_f_s1}. Skipping refinement.")
                                 refinement_failures += 1
                                 master_progress.update(refine_task, advance=1)
                                 master_progress.update(job_task, advance=1)
                                 continue # Skip to next file

                             # Update progress bar description
                             master_progress.update(job_task, description=f"[cyan]Refining: File {idx_r+1}/{len(successfully_transcribed_files)}")
                             master_progress.update(refine_task, description=f"[magenta]File {idx_r+1}/{len(successfully_transcribed_files)}: {os.path.basename(transcript_f_s1)}")

                             # Call refinement function with Stage 1 input path and Stage 2 target path
                             if process_single_file_refinement(llm_api, transcript_f_s1, target_output_path_s2, prompt_template):
                                 refinement_successes += 1
                             else:
                                 refinement_failures += 1
                                 logging.warning(f"Failed refinement for: {transcript_f_s1} -> {target_output_path_s2}")
                             master_progress.update(refine_task, advance=1)
                             master_progress.update(job_task, advance=1) # Advance overall progress after refinement step

                        # Stop the refinement task explicitly INSIDE the Live context
                        if 'refine_task' in locals() and refine_task is not None:
                            try:
                                # No need to check if task exists in list, just update/stop
                                master_progress.update(refine_task, description="[magenta]Refinement Phase Complete")
                                master_progress.stop_task(refine_task)
                            except Exception as e: # Catch potential errors if task is already gone or invalid
                                logging.debug(f"Could not stop refine task {refine_task} (may already be stopped/removed): {e}")


                    phase2_summary = f"Phase 2: LLM Refinement Complete - Successfully refined: {refinement_successes} transcript(s), Failed or skipped: {refinement_failures} transcript(s)"
                    logging.info(phase2_summary)
                    # Print summary outside the live context
                    console.print("\n--- Phase 2: LLM Refinement Complete ---")
                    console.print(f"  Successfully refined: {refinement_successes} transcript(s)")
                    console.print(f"  Failed or skipped:    {refinement_failures} transcript(s)")
            else:
                # Client initialization or warm-up failed earlier
                logging.error("LLM Client initialization or warm-up failed. Skipping Phase 2 (Refinement).")
                console.print("\n[error]LLM Client initialization or warm-up failed. Skipping Phase 2 (Refinement). Raw transcripts have been saved.[/error]", style="error")

    elif Config.TRANSCRIPT_MODE != 'llm':
        logging.info("Skipping Phase 2: LLM Refinement (TRANSCRIPT_MODE is 'direct').")
        console.print("\nSkipping Phase 2: LLM Refinement (TRANSCRIPT_MODE is 'direct').")
    elif pipeline_failed:
        logging.warning("Skipping Phase 2: LLM Refinement due to fatal error during transcription phase.")
        console.print("\nSkipping Phase 2: LLM Refinement due to fatal error during transcription phase.")

    # Finalize progress tracking outside the Live context if it was used for refinement
    # Ensure the job task is marked complete
    master_progress.update(job_task, description="[bold cyan]Overall Process Complete", completed=len(files_to_process_phase1)*2)
    if not master_progress.tasks[job_task].finished:
         master_progress.stop_task(job_task)

    # Final Summary (printed directly to console, not Live)
    logging.info("Processing finished.")

    # Create summary data
    summary_data = {
        "Transcriptions succeeded": len(successfully_transcribed_files),
        "Transcriptions failed": transcription_failures
    }
    if Config.TRANSCRIPT_MODE == 'llm':
        summary_data.update({
            "Refinements succeeded": refinement_successes,
            "Refinements failed": refinement_failures
        })

    # Use ui.summary_table for consistency
    summary_table = ui.summary_table(summary_data)
    # Use ui.create_panel for the final summary panel
    console.print(ui.create_panel(summary_table,
                       title="[bold]Processing Complete[/bold]",
                       subtitle=f"Completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                       border_style="green"))

    # Defensive cleanup - runs regardless of success/failure
    # Prevents resource leaks in all scenarios
    release_model_and_processor()  # Ensure release if script exits unexpectedly

if __name__ == "__main__":
    # Using spawn is apparently safer with CUDA
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        try:
            multiprocessing.set_start_method("spawn", force=True)
            logging.debug("Set multiprocessing start method to 'spawn'.")
        except RuntimeError as e:
            # This might happen if context is already set, e.g., in some environments
            logging.warning(f"Could not force multiprocessing start method to 'spawn': {e}. Using default: {multiprocessing.get_start_method()}")

    run()
