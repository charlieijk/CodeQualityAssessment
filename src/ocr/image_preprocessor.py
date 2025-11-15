import cv2
import numpy as np
from PIL import Image
import pytesseract


class ImagePreprocessor:
    def __init__(self):
        self.tesseract_config = '--oem 3 --psm 6'

    def preprocess_image(self, image_path):
        """
        Preprocess image for better OCR results
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image from {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply preprocessing steps
        processed = self._enhance_image(gray)

        return processed

    def _enhance_image(self, gray_image):
        """
        Apply various enhancement techniques to improve OCR accuracy
        """
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray_image)

        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(denoised)

        # Morphological operations to clean up
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(contrast_enhanced, cv2.MORPH_CLOSE, kernel)

        # Threshold to binary
        _, binary = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def extract_text(self, processed_image):
        """
        Extract text from preprocessed image using OCR
        """
        # Convert to PIL Image for pytesseract
        pil_image = Image.fromarray(processed_image)

        # Extract text
        text = pytesseract.image_to_string(pil_image, config=self.tesseract_config)

        return text.strip()

    def extract_text_with_confidence(self, processed_image):
        """
        Extract text with confidence scores
        """
        pil_image = Image.fromarray(processed_image)

        # Get detailed data including confidence scores
        data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)

        # Filter out low confidence text
        confident_text = []
        for i, conf in enumerate(data['conf']):
            if int(conf) > 50:  # Only keep text with >50% confidence
                text = data['text'][i].strip()
                if text:
                    confident_text.append(text)

        return ' '.join(confident_text)

    def detect_code_regions(self, image_path):
        """
        Detect potential code regions in the image
        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect horizontal lines (common in code editors)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)

        # Find contours of potential code blocks
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        code_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and h > 20:  # Filter minimum size
                code_regions.append((x, y, w, h))

        return code_regions

    def process_code_image(self, image_path):
        """
        Complete pipeline to process a code screenshot
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)

            # Extract text with confidence
            extracted_text = self.extract_text_with_confidence(processed_image)

            # Detect code regions
            code_regions = self.detect_code_regions(image_path)

            return {
                'extracted_text': extracted_text,
                'code_regions': code_regions,
                'success': True
            }

        except Exception as e:
            return {
                'extracted_text': '',
                'code_regions': [],
                'success': False,
                'error': str(e)
            }
