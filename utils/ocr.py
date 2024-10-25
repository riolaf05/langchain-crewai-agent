from openai import OpenAI
from IPython.display import display
from IPython.display import Image as ip_image
import requests
import base64
import requests
import logging
import pytesseract
from PIL import Image

class OCR:
    #See https://gist.github.com/nealcaren/4ba5f2624baaf5e3ba8fa26e7486f74f
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.client =  OpenAI(max_retries=3,
                              timeout=20.0,)

    # Function to extract text from an image
    def extract_text_from_image(self, image_path):
        # Open the image file
        with Image.open(image_path) as img:
            # Use Tesseract to do OCR on the image
            text = pytesseract.image_to_string(img)

        return text
    
    def extract_text_from_image_url(self, image_url):
        response = self.client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[

            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this article into markdown."},
                    {
                        "type": "image_url",
                        "image_url": image_url,
                    },

                ],
            }
            ],
            max_tokens=1500,
        )
        return response.choices[0].message.content