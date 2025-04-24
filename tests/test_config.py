import pytest
import sys
import os

# Add the project root to the Python path to allow importing 'config'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config import BaseConfig, TranscriptionConfig, OrthographyConfig

def test_config_hierarchy():
    """Verify that the specific config classes inherit from BaseConfig."""
    assert issubclass(TranscriptionConfig, BaseConfig), "TranscriptionConfig should inherit from BaseConfig"
    assert issubclass(OrthographyConfig, BaseConfig), "OrthographyConfig should inherit from BaseConfig"

def test_base_config_defaults():
    config = BaseConfig()
    assert config.LLM_API_BASE_URL == 'http://localhost:11434/v1'
    assert config.FRAGMENT_SIZE == 8000
    assert config.CHUNK_OVERLAP == 500

def test_transcription_config_defaults():
    config = TranscriptionConfig()
    assert config.TRANSCRIBE_MODEL_NAME == 'openai/whisper-medium'
    assert config.AUDIO_BATCH_SIZE == 4

def test_orthography_config_defaults():
    config = OrthographyConfig()
    assert config.LLM_TIMEOUT == 120
    assert config.LLM_RETRIES == 3

def test_get_env_variable_handling():
    # Test bool handling
    os.environ['TEST_BOOL'] = 'true'
    assert BaseConfig.get_env('TEST_BOOL', False, bool) is True
    
    # Test int handling
    os.environ['TEST_INT'] = '123'
    assert BaseConfig.get_env('TEST_INT', 0, int) == 123
    
    # Test float handling
    os.environ['TEST_FLOAT'] = '123.45'
    assert BaseConfig.get_env('TEST_FLOAT', 0.0, float) == 123.45
    
    # Test str handling
    os.environ['TEST_STR'] = 'test_string'
    assert BaseConfig.get_env('TEST_STR', '', str) == 'test_string'
    
    # Test default value
    assert BaseConfig.get_env('NON_EXISTENT_VAR', 'default_value', str) == 'default_value'

def test_config_validation():
    # Test CHUNK_OVERLAP validation
    os.environ['CHUNK_OVERLAP'] = '9000'
    os.environ['FRAGMENT_SIZE'] = '8000'
    with pytest.raises(ValueError):
        BaseConfig()
    
    # Reset environment variables
    del os.environ['CHUNK_OVERLAP']
    del os.environ['FRAGMENT_SIZE']

def test_audio_config_defaults():
    config = AudioConfig()
    assert config.TARGET_SAMPLE_RATE == 16000
    assert config.DEFAULT_AUDIO_FORMAT == 'wav'