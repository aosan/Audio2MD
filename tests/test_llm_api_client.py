import unittest
from unittest.mock import patch, MagicMock
from llm_api_client import LLMApiClient
from config import TranscriptionConfig

class TestLLMApiClient(unittest.TestCase):

    @patch('llm_api_client.TranscriptionConfig')
    def test_init(self, mock_config):
        # Correctly configure mock_config to have required attributes
        mock_config_instance = mock_config.return_value
        mock_config_instance.LLM_API_BASE_URL = 'http://test-url.com'
        mock_config_instance.LLM_API_KEY = 'test_key'
        mock_config_instance.LLM_TIMEOUT = 30
        mock_config_instance.LLM_RETRIES = 2
        mock_config_instance.LLM_MODEL_NAME = 'test_model'
        mock_config_instance.LLM_TEMPERATURE = None
        
        client = LLMApiClient(mock_config_instance)
        self.assertIsNotNone(client)

    @patch('llm_api_client.LLMApiClient._initialize_client')
    def test_warm_up(self, mock_initialize_client):
        mock_config = MagicMock(spec=TranscriptionConfig)
        mock_config.LLM_MODEL_NAME = 'test_model'
        client = LLMApiClient(mock_config)
        client.client = MagicMock()
        client.client.models.list = MagicMock(return_value=True)
        result = client.warm_up()
        self.assertTrue(result)

    @patch('llm_api_client.LLMApiClient._initialize_client')
    def test_process_text(self, mock_initialize_client):
        mock_config = MagicMock(spec=TranscriptionConfig)
        mock_config.LLM_MODEL_NAME = 'test_model'
        mock_config.LLM_TEMPERATURE = None
        client = LLMApiClient(mock_config)
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        client.client = MagicMock()
        client.client.chat.completions.create = MagicMock(return_value=mock_response)
        
        result = client.process_text("Test prompt")
        self.assertEqual(result, "Test response")

if __name__ == '__main__':
    unittest.main()