import os
import json
import cv2
import numpy as np
from ocr_processor import OCRProcessor
from gemini_ocr_processor import GeminiOCRProcessor
from data_parser import DataParser
from app import preprocess_image

class DualOCRProcessor:
    """A processor that combines results from both text-optimized and handwriting-optimized OCR."""
    
    def __init__(self, ocr_processor, data_parser):
        """Initialize with existing OCR processor and data parser instances."""
        self.ocr_processor = ocr_processor
        self.data_parser = data_parser
    
    def process_image(self, image_path):
        """Process an image using dual-mode OCR and combine the results.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Combined data extracted from the image
        """
        # Process the image in dual mode to get both text and handwriting optimized versions
        processed_images = preprocess_image(image_path, enhance_mode='dual')
        
        if processed_images is None:
            # If preprocessing failed, try with the original image
            return self._process_with_original(image_path)
        
        text_optimized_image, handwriting_optimized_image = processed_images
        
        # Create temporary files for the processed images
        text_temp_path = f"{image_path}_text_processed.jpg"
        handwriting_temp_path = f"{image_path}_handwriting_processed.jpg"
        
        try:
            # Save the processed images to temporary files
            cv2.imwrite(text_temp_path, text_optimized_image)
            cv2.imwrite(handwriting_temp_path, handwriting_optimized_image)
            
            # Process with text-optimized image (good for printed text, account numbers, dates, amounts)
            text_optimized_text = self.ocr_processor.detect_text(text_temp_path)
            text_optimized_data = self.data_parser.parse_deposit_slip(text_optimized_text)
            
            # Process with handwriting-optimized image (good for special markers like HPWIN/HPWINVIP)
            handwriting_optimized_text = self.ocr_processor.detect_text(handwriting_temp_path)
            handwriting_optimized_data = self.data_parser.parse_deposit_slip(handwriting_optimized_text)
            
            # Combine the results, prioritizing the most reliable source for each field
            combined_data = self._combine_results(text_optimized_data, handwriting_optimized_data)
            
            return combined_data
            
        finally:
            # Clean up temporary files
            if os.path.exists(text_temp_path):
                os.unlink(text_temp_path)
            if os.path.exists(handwriting_temp_path):
                os.unlink(handwriting_temp_path)
    
    def _process_with_original(self, image_path):
        """Fallback to processing with the original image if preprocessing fails."""
        text = self.ocr_processor.detect_text(image_path)
        return self.data_parser.parse_deposit_slip(text)
    
    def _combine_results(self, text_data, handwriting_data):
        """Combine results from text-optimized and handwriting-optimized processing.
        
        Prioritize the most reliable source for each field:
        - Text optimization is better for account numbers, dates, amounts
        - Handwriting optimization is better for special markers
        """
        combined_data = {}
        
        # Start with text-optimized data as the base
        combined_data.update(text_data)
        
        # Override with handwriting-optimized data for special text detection
        # Always prioritize handwriting data for special text detection
        if handwriting_data.get('has_special_text', False):
            combined_data['has_special_text'] = True
            combined_data['special_text_found'] = handwriting_data.get('special_text_found')
            if 'special_text_match' in handwriting_data:
                combined_data['special_text_match'] = handwriting_data.get('special_text_match')
            print(f"Special text found in handwriting-optimized image: {handwriting_data.get('special_text_found')}")
        # Even if text data has special text but handwriting doesn't, keep it
        elif text_data.get('has_special_text', False):
            combined_data['has_special_text'] = True
            combined_data['special_text_found'] = text_data.get('special_text_found')
            if 'special_text_match' in text_data:
                combined_data['special_text_match'] = text_data.get('special_text_match')
            print(f"Special text found in text-optimized image: {text_data.get('special_text_found')}")
        else:
            # No special text found in either image
            combined_data['has_special_text'] = False
            combined_data['special_text_found'] = None
            print("No special text found in either image")
            
            # One last check - look for special text patterns in the raw text from both images
            text_optimized_text = text_data.get('raw_text', '')
            handwriting_optimized_text = handwriting_data.get('raw_text', '')
            combined_text = text_optimized_text + " " + handwriting_optimized_text
            
            special_text = self._check_for_special_text(combined_text)
            if special_text:
                combined_data['has_special_text'] = True
                combined_data['special_text_found'] = special_text
                print(f"Special text found in combined raw text analysis: {special_text}")
        
        # For other fields, use the most complete/reliable data
        # If text_data has a field but handwriting_data has a more complete version, use that
        for field in ['account_number', 'date', 'amount', 'transaction_type']:
            # For account numbers, prefer the longer one as it's likely more complete
            if field == 'account_number':
                text_account = text_data.get(field, '')
                handwriting_account = handwriting_data.get(field, '')
                if handwriting_account and (not text_account or len(handwriting_account) > len(text_account)):
                    combined_data[field] = handwriting_account
            
            # For transaction type, if one is UNKNOWN but the other isn't, use the known one
            elif field == 'transaction_type':
                if text_data.get(field) == 'UNKNOWN' and handwriting_data.get(field) != 'UNKNOWN':
                    combined_data[field] = handwriting_data.get(field)
            
            # For other fields, if text_data doesn't have it but handwriting_data does, use handwriting
            elif field not in text_data and field in handwriting_data:
                combined_data[field] = handwriting_data.get(field)
        
        return combined_data
    
    def process_with_gemini(self, image_path):
        """Process with Gemini if available, otherwise fall back to dual processing."""
        if isinstance(self.ocr_processor, GeminiOCRProcessor):
            try:
                # Try to get structured data directly from Gemini
                json_response = self.ocr_processor.extract_structured_data(image_path)
                # Parse the JSON string to a dictionary
                try:
                    slip_data = json.loads(json_response)
                    
                    # Validate and ensure all required fields are present
                    if 'transaction_type' not in slip_data:
                        slip_data['transaction_type'] = 'UNKNOWN'
                    
                    if 'has_special_text' not in slip_data:
                        slip_data['has_special_text'] = False
                    
                    if 'special_text_found' not in slip_data:
                        slip_data['special_text_found'] = None
                    
                    # Ensure special_text_found is properly set based on has_special_text
                    if slip_data['has_special_text'] and not slip_data['special_text_found']:
                        # If has_special_text is True but no specific text is identified, default to HPWIN
                        slip_data['special_text_found'] = 'HPWIN'
                    elif not slip_data['has_special_text'] and slip_data['special_text_found']:
                        # If special_text_found has a value but has_special_text is False, correct it
                        slip_data['has_special_text'] = True
                        
                    # Double-check special text detection with a secondary method
                    # This helps catch cases where the model might have missed the special text
                    if not slip_data['has_special_text']:
                        # Get raw text from the image as a backup check
                        raw_text = self.ocr_processor.detect_text(image_path)
                        # Check for special text patterns in the raw text
                        special_text = self._check_for_special_text(raw_text)
                        if special_text:
                            slip_data['has_special_text'] = True
                            slip_data['special_text_found'] = special_text
                            print(f"Special text found in secondary check of raw text: {special_text}")
                    
                    return slip_data
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
                    print(f"Problematic JSON: {json_response}")
                    # If JSON parsing fails, fall back to dual processing
                    return self.process_image(image_path)
            except Exception as e:
                print(f"Gemini processing error: {e}")
                # Fall back to dual processing
                return self.process_image(image_path)
        else:
            # Not using Gemini, use dual processing
            return self.process_image(image_path)
            
    def _check_for_special_text(self, text):
        """Check for special text patterns (HPWIN/HPWINVIP) in the given text.
        
        Args:
            text: The text to check for special patterns
            
        Returns:
            String with the detected special text ('HPWINVIP' or 'HPWIN') or None if not found
        """
        import re
        
        # Skip if text is None or empty
        if not text:
            return None
            
        # Preprocess text to handle potential line breaks
        # Replace line breaks with spaces to catch text split across lines
        processed_text = re.sub(r'\s*\n\s*', ' ', text)
        
        # Define patterns to look for in the text
        special_patterns = [
            # HPWINVIP patterns (prioritized first)
            r'(?i)hpwinvip',  # Exact match for HPWINVIP
            r'(?i)hp\s*win\s*vip',  # With spaces: HP WIN VIP
            r'(?i)hpwin\s*vip',  # HPWIN VIP
            r'(?i)h\s*p\s*w\s*i\s*n\s*v\s*i\s*p',  # Spaced out all letters
            r'(?i)h[\s\-]*p[\s\-]*w[\s\-]*i[\s\-]*n[\s\-]*v[\s\-]*i[\s\-]*p',  # With spaces/hyphens
            r'(?i)hpw[1il]nv[1il]p',  # Common substitutions
            r'(?i)hpvv[1il]nv[1il]p',  # 'w' misread as 'vv'
            r'(?i)[hn]pw[1il]nv[1il]p',  # 'h' misread as 'n'
            r'(?i)hpw[1il][hn]v[1il]p',  # 'n' misread as 'h'
            r'(?i)hpw[1il]mv[1il]p',  # 'n' misread as 'm'
            r'(?i)hp.*win.*vip',  # Partial matches with characters between
            r'(?i)h.*p.*w.*i.*n.*v.*i.*p',  # Very loose pattern for distorted text
            r'(?i)hpwn\s*vip',  # Missing 'i' in HPWIN
            r'(?i)hp\s*win\s*vp',  # Missing 'i' in VIP
            r'(?i)hpwn\s*vp',  # Missing 'i' in both HPWIN and VIP
            
            # Standard HPWIN patterns
            r'(?i)hpwin',  # Basic HPWIN
            r'(?i)hp\s*win',  # With spaces
            r'(?i)h\s*p\s*w\s*i\s*n',  # Spaced out letters
            r'(?i)h[\s\-]*p[\s\-]*w[\s\-]*i[\s\-]*n',  # With spaces or hyphens
            r'(?i)hpw[1il]n',  # Common substitutions for 'i'
            r'(?i)hpvv[1il]n',  # 'w' misread as 'vv'
            r'(?i)[hn]pw[1il]n',  # 'h' misread as 'n'
            r'(?i)hpw[1il][hn]',  # 'n' misread as 'h'
            r'(?i)hpw[1il]m',  # 'n' misread as 'm'
            r'(?i)hp.*win',  # Partial matches with characters between
            r'(?i)h.*p.*w.*i.*n',  # Very loose pattern for highly distorted text
            r'(?i)hpwn',  # Missing 'i' in HPWIN
            
            # Line-break detection patterns
            r'(?i)hp\s*$',  # HP at end of line
            r'(?i)^\s*win',  # WIN at start of line
            r'(?i)hpw\s*$',  # HPW at end of line
            r'(?i)^\s*in',  # IN at start of line
            r'(?i)h\s*$',  # H at end of line
            r'(?i)^\s*p',  # P at start of line
            r'(?i)w\s*$',  # W at end of line
            r'(?i)^\s*i',  # I at start of line
            r'(?i)n\s*$',  # N at end of line
            r'(?i)^\s*vip',  # VIP at start of line
            r'(?i)hpwin\s*$',  # HPWIN at end of line
            r'(?i)^\s*v',  # V at start of line (for VIP)
            
            # Account number adjacent patterns
            r'(?i)\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{0,4}[\s\-]*hp',  # Account number followed by HP
            r'(?i)\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{0,4}[\s\-]*hpw',  # Account number followed by HPW
            r'(?i)\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{0,4}[\s\-]*hpwin',  # Account number followed by HPWIN
            r'(?i)hp[\s\-]*\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{0,4}',  # HP before account number
            r'(?i)hpwin[\s\-]*\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{0,4}'  # HPWIN before account number
        ]
        
        # Check for HPWINVIP patterns first (higher priority)
        for i, pattern in enumerate(special_patterns):
            # The first set of patterns are for HPWINVIP (see the pattern definitions above)
            if i < 15 and re.search(pattern, processed_text):
                return 'HPWINVIP'
                
        # Check for HPWINVIP in original text
        for i, pattern in enumerate(special_patterns):
            if i < 15 and re.search(pattern, text):
                return 'HPWINVIP'
                
        # Then check for HPWIN patterns
        for i, pattern in enumerate(special_patterns):
            if i >= 15 and re.search(pattern, processed_text):
                return 'HPWIN'
                
        # Check for HPWIN in original text
        for i, pattern in enumerate(special_patterns):
            if i >= 15 and re.search(pattern, text):
                return 'HPWIN'
                
        return None