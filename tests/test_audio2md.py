import unittest
from unittest.mock import patch, MagicMock
import audio2md
import os
import logging
import torch
import numpy as np

class TestAudio2MD(unittest.TestCase):

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.backends.mps.is_available', return_value=False)
    def test_detect_device_and_dtype_cuda_available(self, mock_mps_is_available, mock_cuda_is_available):
        device, dtype, backend, gpu_name = audio2md.detect_device_and_dtype()
        self.assertEqual(device, "cuda:0")
        self.assertEqual(backend, "CUDA")
        self.assertEqual(dtype, torch.float16)
        self.assertIsNotNone(gpu_name)

    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=True)
    def test_detect_device_and_dtype_mps_available(self, mock_mps_is_available, mock_cuda_is_available):
        device, dtype, backend, gpu_name = audio2md.detect_device_and_dtype()
        self.assertEqual(device, "mps")
        self.assertEqual(backend, "Apple Metal (MPS)")
        self.assertEqual(dtype, torch.float16)
        self.assertEqual(gpu_name, "Apple Silicon")

    @patch('torchaudio.load')
    def test_load_and_preprocess_audio(self, mock_torchaudio_load):
        mock_waveform = torch.tensor([[1, 2, 3]])  # Mock stereo audio
        mock_sr = 44100
        mock_torchaudio_load.return_value = (mock_waveform, mock_sr)

        with patch('torchaudio.transforms.Resample') as mock_resample:
            mock_resampler = MagicMock()
            mock_resample.return_value = mock_resampler
            mock_resampler.return_value = torch.tensor([1, 2, 3])  # Mock resampled waveform

            audio_array, sr = audio2md.load_and_preprocess_audio("dummy_path.wav")
            self.assertIsInstance(audio_array, np.ndarray)
            self.assertEqual(sr, audio2md.Audio_Config.TARGET_SAMPLE_RATE)


    @patch('audio2md.AutoModelForSpeechSeq2Seq.from_pretrained')
    @patch('audio2md.AutoProcessor.from_pretrained')
    def test_get_model_and_processor(self, mock_processor_from_pretrained, mock_model_from_pretrained):
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_from_pretrained.return_value = mock_model
        mock_processor_from_pretrained.return_value = mock_processor
        
        model, processor = audio2md.get_model_and_processor()
        self.assertEqual(model, mock_model)
        self.assertEqual(processor, mock_processor)

    def test_release_model_and_processor(self):
        # Mock _MODEL_AND_PROCESSOR
        mock_model = MagicMock()
        mock_processor = MagicMock()
        audio2md._MODEL_AND_PROCESSOR = (mock_model, mock_processor)
        
        audio2md.release_model_and_processor()
        self.assertIsNone(audio2md._MODEL_AND_PROCESSOR)

if __name__ == '__main__':
    unittest.main()