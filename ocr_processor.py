from google.cloud import vision
import re

class OCRProcessor:
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()
        # Common patterns for post-processing
        self.account_pattern = r'(?:\d{4}[- ]?){2,5}\d{1,4}|\d{10,18}'  # Bank account pattern
        self.amount_pattern = r'\$?\d+[,.]\d{2}'  # Money amount pattern
        self.date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'  # Date pattern
    
    def detect_text(self, image_path):
        """Detects text in the image using Google Cloud Vision API with enhanced settings."""
        with open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        
        # Create feature object with more specific settings
        features = [
            # Use document_text_detection for better handling of both printed and handwritten text
            vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION),
            # Also add text detection which sometimes works better for certain formats
            vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION)
        ]
        
        # Create request with image and features
        request = vision.AnnotateImageRequest(image=image, features=features)
        response = self.client.batch_annotate_images(requests=[request])
        
        # Check for errors
        if response.responses[0].error.message:
            raise Exception(
                '{} For more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(
                    response.responses[0].error.message))
        
        # Get text from document text detection (primary method)
        doc_text = ''
        if response.responses[0].full_text_annotation.text:
            doc_text = response.responses[0].full_text_annotation.text
        
        # Get text from text detection (backup method)
        text_annotations = response.responses[0].text_annotations
        text_detection = text_annotations[0].description if text_annotations else ''
        
        # Use the better result (usually the one with more text)
        extracted_text = doc_text if len(doc_text) >= len(text_detection) else text_detection
        
        # Apply post-processing to improve the extracted text
        return self._post_process_text(extracted_text)
    
    def _post_process_text(self, text):
        """Apply post-processing to improve OCR results."""
        if not text:
            return ''
            
        # Convert to lowercase for processing but keep original for return
        text_lower = text.lower()
        processed_text = text
        
        # Fix common OCR errors
        replacements = {
            # Common OCR errors for financial documents
            'o': '0',  # Letter 'o' to number '0' in numeric contexts
            'l': '1',  # Letter 'l' to number '1' in numeric contexts
            'i': '1',  # Letter 'i' to number '1' in numeric contexts
            's': '5',  # Letter 's' to number '5' in numeric contexts
            'b': '6',  # Letter 'b' to number '6' in numeric contexts
            'g': '9',  # Letter 'g' to number '9' in numeric contexts
        }
        
        # Find potential account numbers and fix them
        account_matches = re.finditer(self.account_pattern, text)
        for match in account_matches:
            account_num = match.group()
            fixed_account = account_num
            
            # Only replace characters in numeric contexts
            for char, replacement in replacements.items():
                # Check if this is in a numeric context (surrounded by numbers)
                #fixed_account = re.sub(f'(?<=\d){char}(?=\d)', replacement, fixed_account)
                fixed_account = re.sub(rf'(?<=\d){char}(?=\d)', replacement, fixed_account)
            
            # Replace in the original text if changes were made
            if fixed_account != account_num:
                processed_text = processed_text.replace(account_num, fixed_account)
        
        # Fix amount formatting
        amount_matches = re.finditer(self.amount_pattern, text)
        for match in amount_matches:
            amount = match.group()
            # Ensure proper decimal formatting
            fixed_amount = re.sub(r'[,\s]', '.', amount)
            if fixed_amount != amount:
                processed_text = processed_text.replace(amount, fixed_amount)
        
        # Enhanced handwritten special text detection (HPWIN, HPWINVIP)
        # More comprehensive pattern matching for handwritten text variations
        # Common OCR errors and variations in handwritten text
        hp_win_patterns = [
            # Basic variations
            r'hpw[il1]n',       # Matches hpwin, hpw1n, hpwln
            r'hpvv[il1]n',      # Matches hpvvin, hpvv1n, hpvvln
            r'hp[\s-]*w[il1]n', # Matches hp win, hp-win with various spacing
            r'h[\s-]*p[\s-]*w[il1]n', # Matches h p w i n with spaces
            r'hpvvin',          # Common misread
            r'hpwn',            # Missing letter
            r'hpwm',            # 'in' misread as 'm'
            r'hpwim',           # 'n' misread as 'm'
            r'npwin',           # 'h' misread as 'n'
            r'hpwih',           # 'n' misread as 'h'
            # Additional variations
            r'h[bp]w[il1]n',     # 'p' misread as 'b'
            r'[hn]pw[il1]n',     # 'h' misread as 'n'
            r'hpw[il1][nm]',     # 'n' misread as 'm'
            r'hpw[il1]\w',       # Any character after 'hpwi'
            r'[nm]pw[il1]n',     # First letter misread
            r'hp\s*w\s*[il1]\s*n', # Spaces between all letters
            r'h\s*p\s*w\s*[il1]\s*n', # Spaces between all letters
            r'hpvv[il1]\w',      # Any character after 'hpwi'
            r'\w{0,2}pw[il1]n',   # First 1-2 chars might be wrong
            r'hpw[il1]\w{0,2}'    # Last 1-2 chars might be wrong
        ]
        
        hp_win_vip_patterns = [
            # Basic variations
            r'hpw[il1]nv[il1]p',  # Matches hpwinvip with various 'i' and 'l' as '1'
            r'hpvv[il1]nv[il1]p', # Matches hpvvinvip with variations
            r'hp[\s-]*w[il1]n[\s-]*v[il1]p', # With spaces or dashes
            r'hpvvinvip',        # Common misread
            r'hpwinv[il1]p',     # Standard with variations in 'vip'
            r'hpw[il1]nvip',     # Standard with variations in 'win'
            # Additional variations
            r'hp\s*w[il1]n\s*v[il1]p', # Spaces between sections
            r'hpw[il1]n\s*v[il1]p',   # Space between win and vip
            r'h\s*p\s*w\s*[il1]\s*n\s*v\s*[il1]\s*p', # Spaces everywhere
            r'hpw[il1]n[-_]?v[il1]p', # Dash or underscore separator
            r'hpw[il1]n\s+v[il1]p',  # Multiple spaces
            r'hp\s*w[il1]n\s*v\s*[il1]\s*p', # Various spacing patterns
            r'[hn]pw[il1]nv[il1]p',  # First letter variation
            r'hpw[il1]nv[il1][pb]',  # Last letter variation
            r'hpw[il1]n\s*v[il1]\w', # Last letter could be anything
            r'\w{0,2}pw[il1]nv[il1]p', # First 1-2 chars might be wrong
            r'hpw[il1]nv[il1]\w{0,2}' # Last 1-2 chars might be wrong
        ]
        
        # Check for HPWINVIP first (more specific)
        for pattern in hp_win_vip_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                # Find the actual match to preserve case in replacement
                match = re.search(pattern, processed_text, re.IGNORECASE)
                if match:
                    processed_text = processed_text.replace(match.group(), 'HPWINVIP')
                    break
        
        # Then check for HPWIN (if HPWINVIP wasn't found)
        if 'HPWINVIP' not in processed_text:
            for pattern in hp_win_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    # Find the actual match to preserve case in replacement
                    match = re.search(pattern, processed_text, re.IGNORECASE)
                    if match:
                        processed_text = processed_text.replace(match.group(), 'HPWIN')
                        break
        
        return processed_text
