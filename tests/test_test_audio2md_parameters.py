import unittest
from unittest.mock import patch, MagicMock
import test_audio2md_parameters
import os
import logging
import numpy as np
import torch

class TestAudio2MDParameters(unittest.TestCase):

    @patch('torchaudio.load')
    def test_load_audio(self, mock_torchaudio_load):
        mock_waveform = torch.tensor([[1, 2, 3]])  # Mock stereo audio
        mock_sr = 44100
        mock_torchaudio_load.return_value = (mock_waveform, mock_sr)

        with patch('torchaudio.transforms.Resample') as mock_resample:
            mock_resampler = MagicMock()
            mock_resample.return_value = mock_resampler
            mock_resampler.return_value = torch.tensor([1, 2, 3])  # Mock resampled waveform

            audio_array, sr = test_audio2md_parameters.load_audio("dummy_path.wav")
            self.assertIsInstance(audio_array, np.ndarray)
            self.assertEqual(sr, test_audio2md_parameters.Config.TARGET_SAMPLE_RATE)

    def test_compute_audio_attributes(self):
        # Mock load_audio to return a known audio array and sample rate
        with patch('test_audio2md_parameters.load_audio') as mock_load_audio:
            mock_load_audio.return_value = (np.array([1, 2, 3]), 44100)
            attrs = test_audio2md_parameters.compute_audio_attributes("dummy_path.wav")
            self.assertIn("duration", attrs)
            self.assertIn("mean_rms", attrs)
            self.assertIn("mean_zcr", attrs)
            self.assertIn("mean_centroid", attrs)

    def test_dynamic_recommendations(self):
        attrs = {
            "mean_rms": 0.01,
            "mean_zcr": 0.2,
            "duration": 1000
        }
        recs = test_audio2md_parameters.dynamic_recommendations(attrs)
        self.assertEqual(recs["TRANSCRIBE_NUM_BEAMS"], 7)
        self.assertEqual(recs["TRANSCRIBE_MAX_LENGTH"], 448)

if __name__ == '__main__':
    unittest.main()