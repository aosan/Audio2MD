import unittest
from unittest.mock import patch, MagicMock
import a2md_orthography
import os
import logging
from config import TranscriptionConfig

class TestA2MDOrthography(unittest.TestCase):

    @patch('ui.core.Audio2MDUI')
    def test_read_prompt_template(self, mock_ui):
        # Create a temporary prompt template file
        with open("temp_prompt.txt", "w") as f:
            f.write("Test prompt template")
        
        mock_console = MagicMock()
        mock_ui.return_value.get_console.return_value = mock_console
        
        content = a2md_orthography.read_prompt_template("temp_prompt.txt", mock_console)
        self.assertEqual(content, "Test prompt template")
        
        # Clean up
        os.remove("temp_prompt.txt")

    @patch('llm_api_client.LLMApiClient')
    def test_refine_transcript_text(self, mock_llm_api_client):
        mock_llm_api = mock_llm_api_client.return_value
        mock_llm_api.process_text.return_value = "Refined transcript text"
        
        transcript_text = "Original transcript text"
        prompt_template = "Part {part_number}/{total_parts}: {chunk}"
        
        refined_text = a2md_orthography.refine_transcript_text(mock_llm_api, transcript_text, prompt_template)
        self.assertEqual(refined_text, "Refined transcript text")

if __name__ == '__main__':
    unittest.main()