import os
import json
import tempfile
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
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

# Configure Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For flash messages

# Configure upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Import image preprocessing function from app.py
from app import preprocess_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was submitted
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # If user submits empty form
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    # Process the file if it's valid
    if file and allowed_file(file.filename):
        # Save the file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            file.save(temp.name)
            temp_path = temp.name
        
        try:
            # Use our dual OCR processor for better results
            from dual_ocr_processor import DualOCRProcessor
            dual_processor = DualOCRProcessor(ocr_processor, data_parser)
            
            # Process with dual mode (both text and handwriting optimizations)
            if use_gemini and isinstance(ocr_processor, GeminiOCRProcessor):
                try:
                    # Try to get structured data directly from Gemini first
                    json_response = ocr_processor.extract_structured_data(temp_path)
                    try:
                        slip_data = json.loads(json_response)
                    except json.JSONDecodeError:
                        # Fall back to dual processing
                        slip_data = dual_processor.process_image(temp_path)
                except Exception as e:
                    print(f"Error with Gemini processing: {e}")
                    # Fall back to dual processing
                    slip_data = dual_processor.process_image(temp_path)
            else:
                # Standard dual processing flow
                slip_data = dual_processor.process_image(temp_path)
            
            # Clean up temporary files
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            # Return the extracted data as JSON
            return jsonify({
                'success': True,
                'data': slip_data
            })
            
        except Exception as e:
            # Clean up temporary files in case of error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    flash('Invalid file type. Please upload a JPG, JPEG, or PNG file.')
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)