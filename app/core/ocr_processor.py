
import pytesseract
from PIL import Image
import pdf2image
from typing import Union, List
import logging

class OCRProcessor:
    def __init__(self, languages: List[str] = ['eng']):
        self.languages = '+'.join(languages)
        
    def process_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        images = pdf2image.convert_from_path(pdf_path)
        text = ""
        for image in images:
            text += self.process_image(image)
        return text
    
    def process_image(self, image: Union[str, Image.Image]) -> str:
        """Extract text from image"""
        if isinstance(image, str):
            image = Image.open(image)
        return pytesseract.image_to_string(image, lang=self.languages)