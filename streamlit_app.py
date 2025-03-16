import os
import json
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from ocr_processor import OCRProcessor
from gemini_ocr_processor import GeminiOCRProcessor
from data_parser import DataParser
from transaction_matcher import TransactionMatcher
from app import preprocess_image

# Load environment variables from .env file if it exists
# In local development, use .env file
# In Streamlit Cloud, use secrets
try:
    load_dotenv()
except:
    pass

# Set page configuration
st.set_page_config(
    page_title="Bank Slip OCR",
    page_icon="🏦",
    layout="wide"
)

# Get credentials from environment variables or Streamlit secrets
# First try to get from secrets (for Streamlit Cloud deployment)
try:
    credentials_path = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    use_gemini = st.secrets.get("USE_GEMINI", "false").lower() == "true"
except Exception:
    # Fall back to environment variables (for local development)
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    use_gemini = os.getenv('USE_GEMINI', 'false').lower() == 'true'

if not credentials_path:
    st.error("GOOGLE_APPLICATION_CREDENTIALS not found in environment variables or secrets")
    st.stop()

# Set credentials path for Google Cloud client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Set Gemini API key if using Gemini
if use_gemini and gemini_api_key:
    os.environ["GEMINI_API_KEY"] = gemini_api_key

@st.cache_resource
def load_processors():
    if use_gemini:
        st.sidebar.info("Using Gemini API for OCR processing")
        ocr_processor = GeminiOCRProcessor()
    else:
        st.sidebar.info("Using Google Cloud Vision API for OCR processing")
        ocr_processor = OCRProcessor()
        
    data_parser = DataParser()
    transaction_matcher = TransactionMatcher()
    
    # Create the dual OCR processor that combines both text and handwriting optimizations
    from dual_ocr_processor import DualOCRProcessor
    dual_ocr_processor = DualOCRProcessor(ocr_processor, data_parser)
    
    return ocr_processor, data_parser, transaction_matcher, dual_ocr_processor

ocr_processor, data_parser, transaction_matcher, dual_ocr_processor = load_processors()

# Configure upload settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize session state for log
if 'log' not in st.session_state:
    st.session_state.log = []

# Initialize session state for batch results
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []

def process_image(uploaded_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp:
        temp.write(uploaded_file.getvalue())
        temp_path = temp.name
    
    try:
        # Use our dual OCR processor to get the best of both text and handwriting optimizations
        with st.spinner("Processing image with dual OCR optimization..."):
            if use_gemini and isinstance(ocr_processor, GeminiOCRProcessor):
                # For Gemini, try structured data first, then fall back to dual processing
                try:
                    # Try to get structured data directly from Gemini
                    slip_data = dual_ocr_processor.process_with_gemini(temp_path)
                except Exception as e:
                    st.error(f"Error with Gemini processing: {e}")
                    # Fall back to dual processing
                    slip_data = dual_ocr_processor.process_image(temp_path)
            else:
                # Standard dual processing flow
                slip_data = dual_ocr_processor.process_image(temp_path)
            
            # Display the processed images (optional)
            processed_images = preprocess_image(temp_path, enhance_mode='dual')
            if processed_images is not None:
                text_optimized, handwriting_optimized = processed_images
                
                # Create a two-column layout for the images
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Text-Optimized")
                    st.image(text_optimized, caption="Optimized for printed text", use_column_width=True)
                with col2:
                    st.subheader("Handwriting-Optimized")
                    st.image(handwriting_optimized, caption="Optimized for handwritten text", use_column_width=True)
        
        # Add to log
        import datetime
        log_entry = {
            'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'filename': uploaded_file.name,
            'transaction_type': slip_data.get('transaction_type', 'UNKNOWN'),
            'account_number': slip_data.get('account_number', 'N/A'),
            'date': slip_data.get('date', 'N/A'),
            'amount': slip_data.get('amount', 'N/A'),
            'status': slip_data.get('status', 'PROCESSED'),
            'special_text': slip_data.get('special_text_found', 'No')
        }
        st.session_state.log.append(log_entry)
        
        return slip_data
        
    except Exception as e:
        # Clean up temporary files in case of error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        if processed_temp_path and os.path.exists(processed_temp_path):
            os.unlink(processed_temp_path)
        
        st.error(f"Error processing image: {e}")
        return None

def process_batch_images(uploaded_files, transaction_file=None, enhancement_mode='dual'):
    """Process multiple images in batch mode"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = (i) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp:
            temp.write(uploaded_file.getvalue())
            temp_path = temp.name
        
        try:
            # Use dual OCR processing for batch processing as well
            if use_gemini and isinstance(ocr_processor, GeminiOCRProcessor):
                # For Gemini, try structured data first, then fall back to dual processing
                try:
                    # Try to get structured data directly from Gemini
                    slip_data = dual_ocr_processor.process_with_gemini(temp_path)
                except Exception as e:
                    # Fall back to dual processing
                    slip_data = dual_ocr_processor.process_image(temp_path)
            else:
                # Standard dual processing flow
                slip_data = dual_ocr_processor.process_image(temp_path)
            
            # For backward compatibility, if user explicitly chooses a different enhancement mode
            if enhancement_mode not in ['dual', 'auto'] and enhancement_mode != 'none':
                # Process with the specified enhancement mode as a fallback
                processed_image = preprocess_image(temp_path, enhance_mode=enhancement_mode)
                
                # Create a temporary file for the processed image if needed
                processed_temp_path = None
                if processed_image is not None:
                    processed_temp_path = f"{temp_path}_processed.jpg"
                    cv2.imwrite(processed_temp_path, processed_image)
                    temp_path = processed_temp_path
                
                # Use our modular components with the specified enhancement mode
                if use_gemini and isinstance(ocr_processor, GeminiOCRProcessor):
                    try:
                        json_response = ocr_processor.extract_structured_data(temp_path)
                        try:
                            slip_data = json.loads(json_response)
                        except json.JSONDecodeError:
                            text = ocr_processor.detect_text(temp_path)
                            slip_data = data_parser.parse_deposit_slip(text)
                    except Exception:
                        text = ocr_processor.detect_text(temp_path)
                        slip_data = data_parser.parse_deposit_slip(text)
                else:
                    text = ocr_processor.detect_text(temp_path)
                    slip_data = data_parser.parse_deposit_slip(text)
            
            # Match with transactions if a file is selected
            match_result = None
            if transaction_file:
                try:
                    match_result = transaction_matcher.match_transaction(slip_data, transaction_file)
                except Exception as e:
                    match_result = {"error": str(e)}
            
            # Add to results
            result = {
                'filename': uploaded_file.name,
                'slip_data': slip_data,
                'match_result': match_result,
                'status': 'Success'
            }
            
            # Add to log
            import datetime
            log_entry = {
                'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'filename': uploaded_file.name,
                'transaction_type': slip_data.get('transaction_type', 'UNKNOWN'),
                'account_number': slip_data.get('account_number', 'N/A'),
                'date': slip_data.get('date', 'N/A'),
                'amount': slip_data.get('amount', 'N/A'),
                'status': 'PROCESSED',
                'special_text': slip_data.get('special_text_found', 'No')
            }
            st.session_state.log.append(log_entry)
            
        except Exception as e:
            # Add error to results
            result = {
                'filename': uploaded_file.name,
                'slip_data': None,
                'match_result': None,
                'status': f'Error: {str(e)}'
            }
            
            # Add to log
            import datetime
            log_entry = {
                'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'filename': uploaded_file.name,
                'transaction_type': 'ERROR',
                'account_number': 'N/A',
                'date': 'N/A',
                'amount': 'N/A',
                'status': f'ERROR: {str(e)}',
                'special_text': 'No'
            }
            st.session_state.log.append(log_entry)
        
        finally:
            # Clean up temporary files
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            if processed_temp_path and os.path.exists(processed_temp_path):
                os.unlink(processed_temp_path)
        
        results.append(result)
    
    # Complete progress
    progress_bar.progress(1.0)
    status_text.text(f"Completed processing {len(uploaded_files)} images")
    
    return results

def match_with_transactions(slip_data, transaction_file):
    try:
        match_result = transaction_matcher.match_transaction(slip_data, transaction_file)
        return match_result
    except Exception as e:
        st.error(f"Error matching transaction: {e}")
        return None

# Main app layout
st.title("Bank Slip OCR")

# Sidebar for settings
st.sidebar.title("Settings")

# Transaction file selection
transaction_files = []
docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs')
if os.path.exists(docs_dir):
    transaction_files = [f for f in os.listdir(docs_dir) if f.endswith('.xlsx')]

selected_transaction_file = None
if transaction_files:
    selected_transaction_file = st.sidebar.selectbox(
        "Select Transaction File for Matching",
        transaction_files,
        index=0 if transaction_files else None
    )
    if selected_transaction_file:
        selected_transaction_file = os.path.join(docs_dir, selected_transaction_file)
else:
    st.sidebar.warning("No transaction files found in the 'docs' directory")

# Enhancement mode selection
enhancement_mode = st.sidebar.selectbox(
    "Image Enhancement Mode",
    ["auto", "text", "receipt", "none"],
    index=0
)

# Clear log button in sidebar
if st.sidebar.button("Clear Processing Log"):
    st.session_state.log = []
    st.sidebar.success("Log cleared!")

# Clear batch results button in sidebar
if st.sidebar.button("Clear Batch Results"):
    st.session_state.batch_results = []
    st.sidebar.success("Batch results cleared!")

# Main content area with tabs
tab1, tab2 = st.tabs(["Single Image Upload", "Batch Upload"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Bank Slip Image")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="single_upload")
        
        if uploaded_file is not None:
            # Display the original image
            st.image(uploaded_file, caption="Original Image", use_column_width=True)
            
            # Process button
            if st.button("Process Image"):
                with st.spinner("Processing image..."):
                    slip_data = process_image(uploaded_file)
                    
                    if slip_data:
                        st.success("Image processed successfully!")
                        
                        # Match with transactions if a file is selected
                        if selected_transaction_file:
                            with st.spinner("Matching with transactions..."):
                                match_result = match_with_transactions(slip_data, selected_transaction_file)
                                if match_result:
                                    st.success("Transaction matched!")
                                else:
                                    st.warning("No matching transaction found.")

    with col2:
        st.subheader("Extracted Data")
        if uploaded_file is not None and 'slip_data' in locals() and slip_data:
            # Display extracted data in a table
            data_df = pd.DataFrame({
                'Field': slip_data.keys(),
                'Value': [str(v) for v in slip_data.values()]
            })
            st.table(data_df)
            
            # Display matched transaction if available
            if 'match_result' in locals() and match_result:
                st.subheader("Matched Transaction")
                match_df = pd.DataFrame({
                    'Field': match_result.keys(),
                    'Value': [str(v) for v in match_result.values()]
                })
                st.table(match_df)

with tab2:
    st.subheader("Batch Upload Bank Slip Images")
    
    # Batch file uploader
    uploaded_files = st.file_uploader(
        "Choose multiple image files", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    # Process batch button
    if uploaded_files and len(uploaded_files) > 0:
        st.write(f"Selected {len(uploaded_files)} files for batch processing")
        
        if st.button("Process Batch"):
            with st.spinner(f"Processing {len(uploaded_files)} images in batch..."):
                # Process all images in batch
                batch_results = process_batch_images(
                    uploaded_files, 
                    selected_transaction_file, 
                    enhancement_mode
                )
                
                # Store results in session state
                st.session_state.batch_results = batch_results
                
                st.success(f"Batch processing completed! Processed {len(batch_results)} images.")
    
    # Display batch results if available
    if st.session_state.batch_results:
        st.subheader("Batch Processing Results")
        
        # Create tabs for each processed image
        result_tabs = st.tabs([f"Image {i+1}: {result['filename']}" for i, result in enumerate(st.session_state.batch_results)])
        
        for i, (result, tab) in enumerate(zip(st.session_state.batch_results, result_tabs)):
            with tab:
                st.write(f"**File:** {result['filename']}")
                st.write(f"**Status:** {result['status']}")
                
                if result['slip_data']:
                    # Display extracted data
                    st.subheader("Extracted Data")
                    data_df = pd.DataFrame({
                        'Field': result['slip_data'].keys(),
                        'Value': [str(v) for v in result['slip_data'].values()]
                    })
                    st.table(data_df)
                    
                    # Display matched transaction if available
                    if result['match_result']:
                        st.subheader("Matched Transaction")
                        match_df = pd.DataFrame({
                            'Field': result['match_result'].keys(),
                            'Value': [str(v) for v in result['match_result'].values()]
                        })
                        st.table(match_df)
        
        # Summary table of all results
        st.subheader("Batch Summary")
        summary_data = []
        for result in st.session_state.batch_results:
            summary_entry = {
                'Filename': result['filename'],
                'Status': result['status'],
                'Transaction Type': result['slip_data'].get('transaction_type', 'N/A') if result['slip_data'] else 'N/A',
                'Account Number': result['slip_data'].get('account_number', 'N/A') if result['slip_data'] else 'N/A',
                'Date': result['slip_data'].get('date', 'N/A') if result['slip_data'] else 'N/A',
                'Amount': result['slip_data'].get('amount', 'N/A') if result['slip_data'] else 'N/A',
                'Special Text': result['slip_data'].get('special_text_found', 'None') if result['slip_data'] else 'N/A',
                'Matched': 'Yes' if result['match_result'] else 'No'
            }
            summary_data.append(summary_entry)
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Export results button
        if st.button("Export Results to CSV"):
            # Convert to CSV
            csv = summary_df.to_csv(index=False)
            # Create download button
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"batch_results_{timestamp}.csv",
                mime="text/csv"
            )

# Processing log section
st.subheader("Processing Log")
if st.session_state.log:
    log_df = pd.DataFrame(st.session_state.log)
    st.dataframe(log_df, use_container_width=True)
else:
    st.info("No processing records yet. Upload and process an image to see the log.")

# Footer
st.markdown("---")
st.markdown("Bank Slip OCR - Powered by Streamlit and Google Cloud Vision/Gemini API")