import unittest
from pyrit.prompt_converter import PDFConverter
from io import BytesIO

class TestPdfConverter(unittest.TestCase):
    
    def setUp(self):
        # Initialize a PdfConverter object with a sample file path (or a dummy path for testing).
        self.converter = PDFConverter("sample.pdf")
    
    def test_load_pdf_valid_path(self):
        # Test loading a PDF from a valid path.
        try:
            self.converter.load_pdf()
        except Exception as e:
            self.fail(f"load_pdf() raised an exception unexpectedly: {e}")
    
    def test_load_pdf_invalid_path(self):
        # Test loading a PDF from an invalid path.
        self.converter.file_path = "invalid_path.pdf"
        with self.assertRaises(FileNotFoundError):
            self.converter.load_pdf()
    
    def test_convert_pdf_format(self):
        # Test the conversion format, for example, converting to PNG.
        self.converter.load_pdf()
        converted_images = self.converter.convert_to_images(format="PNG")
        
        # Check that images are returned and they have the correct format.
        self.assertIsInstance(converted_images, list)
        for image in converted_images:
            self.assertTrue(image.format == "PNG", "Image format should be PNG")
    
    def test_save_pdf(self):
        # Test saving the converted file to a specific path.
        self.converter.load_pdf()
        converted_images = self.converter.convert_to_images()
        
        output_path = "output_folder/converted_page.png"
        self.converter.save_images(converted_images, output_path)
        
        # Check if the saved image file exists.
        import os
        self.assertTrue(os.path.exists(output_path), "The converted image should be saved to the output path")
    
    def test_edge_case_empty_pdf(self):
        # Test conversion for an empty PDF or a PDF with no pages.
        self.converter.file_path = "empty.pdf"  # Set an empty or invalid PDF file path.
        with self.assertRaises(ValueError):
            self.converter.convert_to_images()
    
    def test_performance_large_pdf(self):
        # Test performance for a large PDF file.
        import time
        self.converter.file_path = "large_sample.pdf"
        self.converter.load_pdf()
        
        start_time = time.time()
        converted_images = self.converter.convert_to_images()
        end_time = time.time()
        
        # Assuming 10 seconds is a reasonable time limit for the conversion
        self.assertLess(end_time - start_time, 10, "Conversion took too long for a large PDF")
    
    def tearDown(self):
        # Any cleanup operations can be added here
        pass

if __name__ == "__main__":
    unittest.main()
