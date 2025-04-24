import unittest
from unittest.mock import patch, MagicMock
from ui.core import Audio2MDUI, LogPanel
import logging
import os

class TestAudio2MDUI(unittest.TestCase):

    @patch('ui.core.LogPanel')
    @patch('ui.core.Console')
    def test_setup_live_logging(self, mock_console, mock_log_panel):
        mock_log_panel_instance = mock_log_panel.return_value
        mock_console_instance = mock_console.return_value
        
        ui = Audio2MDUI()
        ui.setup_live_logging(mock_log_panel_instance, True, "test_log")
        
        # Check if file handler is created when debug_mode is True
        self.assertTrue(any(isinstance(handler, logging.FileHandler) for handler in logging.getLogger().handlers))
        
        # Check if RichHandler is configured correctly
        rich_handler = next((h for h in logging.getLogger().handlers if isinstance(h, logging.Handler)), None)
        self.assertIsNotNone(rich_handler)

if __name__ == '__main__':
    unittest.main()