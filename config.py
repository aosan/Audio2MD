import os
import logging
from typing import Optional, Union
from accelerate import Accelerator
import torch
import sys

class BaseConfig:
    """Central configuration for audio processing components"""
    
    def __init__(self):
        # Core LLM Configuration
        self.LLM_API_BASE_URL = self.get_env('LLM_API_BASE_URL', 'http://localhost:11434/v1', str)
        self.LLM_API_KEY = self.get_env('LLM_API_KEY', None, str)
        self.LLM_MODEL_NAME = self.get_env('LLM_MODEL_NAME', 'mistral', str)
        
        self.LLM_TIMEOUT = self.get_env('LLM_TIMEOUT', 120, int)
        self.LLM_RETRIES = self.get_env('LLM_RETRIES', 3, int)
        self.LLM_TEMPERATURE = self.get_env('LLM_TEMPERATURE', None, float)
        
        # Audio/Text Processing
        self.TARGET_SAMPLE_RATE = self.get_env('TARGET_SAMPLE_RATE', 16000, int)
        self.FRAGMENT_SIZE = self.get_env('FRAGMENT_SIZE', 8000, int)
        self.CHUNK_OVERLAP = self.get_env('CHUNK_OVERLAP', 500, int)
        self.TRANSCRIPT_PROMPT = self.get_env('TRANSCRIPT_PROMPT', 'prompts/transcript_prompt.txt', str)
        self.DEBUG = self.get_env('DEBUG', False, bool)
        self.CUDA_LAUNCH_BLOCKING = self.get_env('CUDA_LAUNCH_BLOCKING', False, bool)
        self.LIBROSA_CACHE_DIR = self.get_env('LIBROSA_CACHE_DIR',
            os.path.join(os.path.dirname(__file__), '.librosa_cache'), str)

        # Validate and ensure cache directory exists
        self._ensure_cache_directory()

        # Validation

    def _ensure_cache_directory(self):
        """Create and validate the cache directory structure"""
        try:
            if os.path.exists(self.LIBROSA_CACHE_DIR) and not os.path.isdir(self.LIBROSA_CACHE_DIR):
                raise RuntimeError(f"LIBROSA_CACHE_DIR path exists as file: {self.LIBROSA_CACHE_DIR}")
            os.makedirs(self.LIBROSA_CACHE_DIR, exist_ok=True)
            os.environ["LIBROSA_CACHE_DIR"] = self.LIBROSA_CACHE_DIR
            logging.debug(f"Configured librosa cache at: {self.LIBROSA_CACHE_DIR}")
        except Exception as e:
            logging.critical(f"Cache directory configuration failed: {e}")
            sys.exit(1)
        self._validate()

    def _validate(self):
        if self.CHUNK_OVERLAP >= self.FRAGMENT_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than FRAGMENT_SIZE")
        if self.FRAGMENT_SIZE < 100:
            raise ValueError("FRAGMENT_SIZE must be at least 100 characters")

    @staticmethod
    def get_env(name: str, default: Optional[Union[str, int, float, bool]], var_type: type):
        """Centralized environment variable handling"""
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
            
        if var_type == bool:
            val = raw_value.strip().lower()
            if val in ('true', '1', 't', 'y', 'yes'):
                return True
            elif val in ('false', '0', 'f', 'n', 'no'):
                return False
            logging.warning(f"Invalid boolean for {name}, using default: {default}")
            return default
            
        try:
            return var_type(raw_value.strip())
        except ValueError:
            logging.error(f"Invalid {name} value, using default: {default}")
            return default

class TranscriptionConfig(BaseConfig):
    """Enhanced transcription configuration with Accelerator support"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize Accelerator with environment configuration
        device_setting = self.get_env('DEVICE', 'auto', str).lower()
        if device_setting not in ('auto', 'gpu', 'cpu'):
            logging.warning(f"Invalid DEVICE setting '{device_setting}'. Using 'auto'")
            device_setting = 'auto'
        self.DEVICE_SETTING = device_setting # Store the validated setting

        self.accelerator = Accelerator(
            cpu=(self.DEVICE_SETTING == 'cpu'),
            device_placement=(self.DEVICE_SETTING == 'auto')
        )
        
        if device_setting == 'gpu' and str(self.accelerator.device) == 'cpu':
            logging.warning("Requested GPU but none available. Using CPU")
            
        self.device = self.accelerator.device
        self.mixed_precision = (
            torch.float16 if self.accelerator.mixed_precision == "fp16" 
            else torch.bfloat16 if self.accelerator.mixed_precision == "bf16" 
            else torch.float32
        )
        
        # Existing configuration parameters
        self.TRANSCRIBE_MODEL_NAME = self.get_env('TRANSCRIBE_MODEL_NAME', 'openai/whisper-medium', str)
        self.CHUNK_LENGTH_S = self.get_env('CHUNK_LENGTH_S', 30, int)
        self.AUDIO_BATCH_SIZE = self.get_env('AUDIO_BATCH_SIZE', 4, int)
        self.SPLIT_AUDIO = self.get_env('SPLIT_AUDIO', 'yes', str).lower()
        self.TRANSCRIPT_MODE = self.get_env('TRANSCRIPT', 'direct', str).lower()
        self.RETRY_TRANSCRIBE = self.get_env('RETRY_TRANSCRIBE', 2, int)
        self.TRANSCRIBE_LANGUAGE = self.get_env('TRANSCRIBE_LANGUAGE', None, str)
        self.MAX_CHUNK_DURATION_S = self.get_env('MAX_CHUNK_DURATION_S', 600, int)
        self.TORCH_DTYPE_OVERRIDE = self.get_env('TORCH_DTYPE_OVERRIDE', 'auto', str).lower() # 'auto', 'float16', 'float32', 'bfloat16'

class OrthographyConfig(BaseConfig):
    """A2MD Orthography specific configuration"""
    
    def __init__(self):
        super().__init__()
        # Specific orthography settings can be added here if needed later

class AudioConfig(BaseConfig):
    """Configuration for audio processing utilities"""
    
    def __init__(self):
        super().__init__()
        self.TARGET_SAMPLE_RATE = self.get_env('TARGET_SAMPLE_RATE', 16000, int)
        self.TRANSCRIBE_NUM_BEAMS = self.get_env('TRANSCRIBE_NUM_BEAMS', 5, int)
        self.TRANSCRIBE_MAX_LENGTH = self.get_env('TRANSCRIBE_MAX_LENGTH', 448, int)