import os
import json
import cv2
import numpy as np
from dotenv import load_dotenv
from ocr_processor import OCRProcessor
from gemini_ocr_processor import GeminiOCRProcessor
from data_parser import DataParser
from transaction_matcher import TransactionMatcher

# Load environment variables from .env file
load_dotenv()

# Get Google Cloud credentials path from environment variable
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if not credentials_path:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set")

# Set credentials path for Google Cloud client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Initialize our processors
# Choose which OCR processor to use (Vision API or Gemini API)
use_gemini = os.getenv('USE_GEMINI', 'false').lower() == 'true'

if use_gemini:
    print("Using Gemini API for OCR processing")
    ocr_processor = GeminiOCRProcessor()
else:
    print("Using Google Cloud Vision API for OCR processing")
    ocr_processor = OCRProcessor()
    
data_parser = DataParser()
transaction_matcher = TransactionMatcher()

def preprocess_image(image_path, enhance_mode='auto'):
    """Preprocess the image to improve OCR accuracy.
    
    Args:
        image_path: Path to the image file
        enhance_mode: 'auto', 'text', 'receipt', 'handwriting', 'dual', or 'none'
    
    Returns:
        Preprocessed image as a numpy array, or None to use original image
        For dual mode, returns a tuple of (text_optimized_image, handwriting_optimized_image)
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if enhance_mode == 'none':
        return None  # Use original image
    
    # For dual mode, process the image with both text and handwriting optimizations
    if enhance_mode == 'dual':
        # Process for printed text
        text_processed = preprocess_for_text(gray)
        
        # Process for handwritten text
        handwriting_processed = preprocess_for_handwriting(gray)
        
        # Return both processed images
        return (text_processed, handwriting_processed)
    
    if enhance_mode == 'auto' or enhance_mode == 'receipt':
        # Apply adaptive thresholding to handle varying lighting conditions
        # This works well for receipts with dark text on light background
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply slight Gaussian blur to reduce noise
        processed = cv2.GaussianBlur(processed, (3, 3), 0)
        
        # Apply morphological operations to enhance text
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        
    elif enhance_mode == 'handwriting':
        processed = preprocess_for_handwriting(gray)
        
    elif enhance_mode == 'text':
        processed = preprocess_for_text(gray)
    
    # Save the processed image temporarily for debugging if needed
    # debug_path = image_path.replace('.', '_processed.')
    # cv2.imwrite(debug_path, processed)
    
    return processed

def preprocess_for_text(gray_image):
    """Optimize image preprocessing for printed text."""
    # Apply bilateral filter to preserve edges while reducing noise
    processed = cv2.bilateralFilter(gray_image, 9, 75, 75)
    
    # Apply Otsu's thresholding
    _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return processed

def preprocess_for_handwriting(gray_image):
    """Optimize image preprocessing for handwritten text."""
    # Apply multiple preprocessing techniques and combine results for better handwriting detection
    
    # Method 1: High contrast enhancement for faint handwriting
    alpha = 3.0  # Increased contrast control (was 2.7)
    beta = 35    # Increased brightness control (was 30)
    contrast_enhanced = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)
    
    # Apply adaptive thresholding with optimized parameters for handwriting
    # Smaller block size (17) for more local adaptivity and lower C value (6) for better sensitivity
    # These parameters help detect text at edges and line breaks better
    processed1 = cv2.adaptiveThreshold(
        contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 6
    )
    
    # Method 2: Edge enhancement for clearer handwriting boundaries
    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(gray_image, 11, 20, 20)  # Adjusted parameters for better edge preservation
    # Apply Canny edge detection with lower thresholds to catch more subtle edges
    # Lower thresholds help detect partial text at line endings
    edges = cv2.Canny(bilateral, 30, 120)  # Lower thresholds to catch more subtle edges
    # Dilate edges to connect broken lines
    kernel_edge = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel_edge, iterations=3)  # Increased iterations
    
    # Method 3: Local histogram equalization to enhance local contrast
    # This helps with detecting handwriting in varying lighting conditions
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray_image)
    _, equalized_thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Combine all three methods
    combined = cv2.bitwise_or(processed1, edges_dilated)
    combined = cv2.bitwise_or(combined, equalized_thresh)
    
    # Apply morphological operations to connect broken strokes in handwriting
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    # Apply dilation to make handwritten text thicker and more visible
    # Increased iterations to better connect characters that might be split
    processed = cv2.dilate(processed, kernel, iterations=4)  # Increased iterations
    
    # Apply additional morphological operations to enhance handwritten text
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    
    # Add border padding to preserve text at edges
    # This helps with text that might be cut off at image boundaries
    processed = cv2.copyMakeBorder(processed, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)  # Increased border
    
    # Invert back to normal polarity (black text on white background)
    processed = cv2.bitwise_not(processed)
    
    return processed

def process_batch(image_folder, transaction_file, enhance_mode='dual'):
    """Processes a batch of deposit slip images and matches with transactions.
    
    Args:
        image_folder: Folder containing images to process
        transaction_file: Excel file with transaction data
        enhance_mode: Image enhancement mode ('dual', 'auto', 'text', 'receipt', 'handwriting', 'none')
    """
    # Create the dual OCR processor that combines both text and handwriting optimizations
    from dual_ocr_processor import DualOCRProcessor
    dual_processor = DualOCRProcessor(ocr_processor, data_parser)
    
    results = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, filename)
            try:
                # Use dual OCR processing by default
                if enhance_mode == 'dual':
                    # Process with dual mode (both text and handwriting optimizations)
                    if use_gemini and isinstance(ocr_processor, GeminiOCRProcessor):
                        try:
                            # Try to get structured data directly from Gemini first
                            slip_data = dual_processor.process_with_gemini(image_path)
                        except Exception as e:
                            print(f"Error with Gemini processing: {e}")
                            # Fall back to dual processing
                            slip_data = dual_processor.process_image(image_path)
                    else:
                        # Standard dual processing flow
                        slip_data = dual_processor.process_image(image_path)
                else:
                    # Legacy mode - use single enhancement mode
                    # Preprocess the image to improve OCR accuracy
                    processed_image = preprocess_image(image_path, enhance_mode=enhance_mode)
                    
                    # Create a temporary file for the processed image if needed
                    temp_image_path = image_path
                    if processed_image is not None:
                        temp_image_path = f"{image_path}_temp_processed.jpg"
                        cv2.imwrite(temp_image_path, processed_image)
                    
                    # Use our modular components
                    if use_gemini and isinstance(ocr_processor, GeminiOCRProcessor):
                        # For Gemini, we can either get raw text or structured data directly
                        try:
                            # Try to get structured data directly from Gemini
                            json_response = ocr_processor.extract_structured_data(temp_image_path)
                            # Parse the JSON string to a dictionary
                            try:
                                slip_data = json.loads(json_response)
                            except json.JSONDecodeError:
                                # If JSON parsing fails, fall back to text extraction and parsing
                                text = ocr_processor.detect_text(temp_image_path)
                                slip_data = data_parser.parse_deposit_slip(text)
                        except Exception as e:
                            print(f"Error with Gemini structured data extraction: {e}")
                            # Fall back to text extraction
                            text = ocr_processor.detect_text(temp_image_path)
                            slip_data = data_parser.parse_deposit_slip(text)
                    else:
                        # Standard Vision API flow
                        text = ocr_processor.detect_text(temp_image_path)
                        slip_data = data_parser.parse_deposit_slip(text)
                    
                    # Clean up temporary file if created
                    if temp_image_path != image_path and os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
                    
                match_result = transaction_matcher.match_transaction(slip_data, transaction_file)
                
                result = {
                    'filename': filename,
                    'extracted_data': slip_data,
                    'matched_transaction': match_result,
                    'enhancement_used': enhance_mode
                }
                results.append(result)
                print(f"Processed {filename}:")
                print(f"Enhancement mode: {enhance_mode}")
                print(f"Extracted data: {slip_data}")
                print(f"Matched transaction: {match_result}\n")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return results

if __name__ == "__main__":
    image_folder = "docs"
    transaction_file = "docs/Bank Update Transaction 1.xlsx"
    # Use enhanced image preprocessing by default
    process_batch(image_folder, transaction_file, enhance_mode='auto')