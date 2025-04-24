import unittest
from processing import determine_file_paths
import os

class TestProcessing(unittest.TestCase):

    def test_determine_file_paths(self):
        # Test with a valid input path
        input_path = "./test_data"
        os.makedirs(input_path, exist_ok=True)
        with open(os.path.join(input_path, "test_file.txt"), "w") as f:
            f.write("Test content")
        
        from unittest.mock import MagicMock
        file_paths, _ = determine_file_paths(input_path, "_test", [".txt"], MagicMock())
        self.assertIn(os.path.join(input_path, "test_file.txt"), file_paths[0])

        # Clean up
        import shutil
        shutil.rmtree(input_path)

if __name__ == '__main__':
    unittest.main()