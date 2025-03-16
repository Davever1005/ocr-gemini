import os
import base64
import re
import json
import google.generativeai as genai
from PIL import Image

class GeminiOCRProcessor:
    def __init__(self):
        # Initialize Gemini API with API key from environment
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        genai.configure(api_key=api_key)
        # Use the Gemini 1.5 Flash model which has multimodal capabilities
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def detect_text(self, image_path):
        """Detects text in the image using Google's Gemini API with enhanced multimodal capabilities."""
        try:
            # Open and prepare the image
            image = Image.open(image_path)
            
            # Create a more detailed prompt that provides context about bank documents
            prompt = (
                "You are a specialized OCR system for bank receipts and deposit slips. Carefully analyze this image and extract the following information:\n\n"
                "1. TRANSACTION TYPE:\n"
                "   - Identify if this is a CDM (Cash Deposit Machine) receipt or ATM_TRANSFER (withdrawal/transfer)\n"
                "   - Look for keywords like 'deposit', 'cash in', 'withdrawal', 'transfer', 'cash out'\n"
                "   - CDM receipts typically show money being added to an account\n"
                "   - ATM_TRANSFER receipts typically show money being withdrawn or transferred\n\n"
                "2. ACCOUNT NUMBER:\n"
                "   - Look for a sequence of 10-18 digits that represents a bank account\n"
                "   - It may be formatted as XXXX-XXXX-XXXX or XXXX XXXX XXXX\n"
                "   - Be careful with OCR errors: 'O' might be '0', 'l' or 'I' might be '1', 'S' might be '5'\n\n"
                "3. DATE:\n"
                "   - Find the transaction date in format DD/MM/YYYY or similar\n"
                "   - Look for labels like 'Date:', 'Transaction Date:', etc.\n\n"
                "4. AMOUNT:\n"
                "   - Find the transaction amount (usually with 2 decimal places)\n"
                "   - Look for currency symbols, or labels like 'Amount:', 'Total:', etc.\n\n"
                "5. REFERENCE NUMBER:\n"
                "   - Look for any reference or transaction ID\n"
                "   - Often labeled as 'Ref:', 'Reference:', 'Transaction ID:', etc.\n\n"
                "6. SPECIAL TEXT (HIGHEST PRIORITY):\n"
                "   - CRITICAL: Your PRIMARY goal is to find 'HPWINVIP' - this is MUCH MORE IMPORTANT than just 'HPWIN'\n"
                "   - ALWAYS prioritize finding and returning 'HPWINVIP' over 'HPWIN' when both might be present\n"
                "   - If you detect ANY indication of 'VIP' after 'HPWIN', you MUST return 'HPWINVIP' not just 'HPWIN'\n"
                "   - Check if the text 'HPWINVIP' or 'HPWIN' appears anywhere in the image\n"
                "   - These are often handwritten and could have OCR errors like 'HPW1NV1P', 'HPVVINVIP', 'HP WIN VIP', 'H P W I N V I P'\n"
                "   - Look for variations with spaces between letters (H P W I N V I P)\n"
                "   - Look for variations with letter substitutions (1 for I, 0 for O, etc.)\n"
                "   - Pay special attention to areas with handwriting that might contain these special markers\n\n"
                "Format the response as plain text with clear labels for each piece of information.\n"
                "If you're uncertain about any field, indicate this rather than guessing."
            )
            
            # Generate content using the image and prompt with temperature setting for more precise extraction
            response = self.model.generate_content(
                [prompt, image],
                generation_config={"temperature": 0.2}  # Lower temperature for more deterministic results
            )
            
            # Return the text response
            return response.text
            
        except Exception as e:
            raise Exception(f"Error processing image with Gemini API: {str(e)}")
    
    def extract_structured_data(self, image_path):
        """Extracts structured data directly using Gemini's understanding of the image."""
        try:
            # Open and prepare the image
            image = Image.open(image_path)
            
            # Create a more detailed prompt with specific instructions for JSON formatting
            prompt = (
                "You are a specialized OCR system for bank receipts. Analyze this image and extract data in this EXACT JSON format:\n\n"
                "```json\n{"
                "\n  \"transaction_type\": string,     // Must be 'CDM' or 'ATM_TRANSFER' or 'UNKNOWN'"
                "\n  \"account_number\": string,      // The bank account number (10-18 digits, may have separators)"
                "\n  \"date\": string,               // Format as YYYY-MM-DD"
                "\n  \"amount\": number,              // Transaction amount as decimal number"
                "\n  \"reference\": string,           // Any reference number/ID or null if none"
                "\n  \"has_special_text\": boolean,   // true if 'HPWIN' or 'HPWINVIP' appears, false otherwise"
                "\n  \"special_text_found\": string   // The exact text found ('HPWIN' or 'HPWINVIP') or null if none"
                "\n}\n```\n\n"
                "IMPORTANT INSTRUCTIONS:\n"
                "1. For TRANSACTION TYPE:\n"
                "   - Look for keywords: 'deposit', 'cash in' = CDM; 'withdrawal', 'transfer', 'cash out' = ATM_TRANSFER\n"
                "   - CDM receipts show money being added; ATM_TRANSFER shows money being withdrawn/transferred\n\n"
                "2. For ACCOUNT NUMBER:\n"
                "   - Fix common OCR errors: 'O'→'0', 'l'/'I'→'1', 'S'→'5', 'B'→'8'\n"
                "   - Remove spaces/dashes in the final output\n\n"
                "3. For DATE:\n"
                "   - Convert any date format to YYYY-MM-DD\n"
                "   - If year is 2-digit, assume 20XX for recent years\n\n"
                "4. For AMOUNT:\n"
                "   - Return as decimal number (e.g., 123.45, not \"$123.45\")\n"
                "   - Fix decimal separators (commas may be misread as periods)\n\n"
                "5. For SPECIAL TEXT (HIGHEST PRIORITY TASK):\n"
                "   - *** THIS IS THE MOST CRITICAL TASK OF THE ENTIRE ANALYSIS ***\n"
                "   - CRITICAL: Your PRIMARY goal is to find 'HPWINVIP' - this is MUCH MORE IMPORTANT than just 'HPWIN'\n"
                "   - ALWAYS prioritize finding and returning 'HPWINVIP' over 'HPWIN' when both might be present\n"
                "   - If you detect ANY indication of 'VIP' after 'HPWIN', you MUST return 'HPWINVIP' not just 'HPWIN'\n"
                "   - Examine EVERY PART of the image, especially handwritten areas, for 'HPWINVIP' or 'HPWIN'\n"
                "   - CRITICAL: Pay special attention to the ACCOUNT/TRANSACTION section of the receipt\n"
                "   - These are HANDWRITTEN markers that may appear ANYWHERE on the receipt/slip\n"
                "   - They often appear at the bottom, in corners, margins, blank spaces, or NEXT TO ACCOUNT NUMBERS\n"
                "   - They may be written in different styles, sizes, and orientations\n"
                "   - They may be partially obscured, faded, or written over other text\n"
                "   - Account for ALL possible OCR errors and variations:\n"
                "     * Letter substitutions: 'HPW1NV1P', 'HPWlNVlP', 'HPVVINVIP', 'NPWINVIP', 'HBWINVIP'\n"
                "     * Spacing variations: 'HP WIN VIP', 'H P W I N V I P', 'H-P-W-I-N-V-I-P'\n"
                "     * Missing letters: 'HPWNVP', 'HWNVP', 'HPINVP'\n"
                "     * Letter merging: 'HPWIMVIP', 'HPWIHVIP', 'HPVVIMVIP'\n"
                "     * For 'HPWIN' without VIP: 'HP WIN', 'HPW1N', 'HPVVIN'\n"
                "     * Common handwriting variations: 'HPWN VIP', 'HP WIN VP', 'HPWN VP'\n"
                "   - Look for PARTIAL matches that could indicate the special text:\n"
                "     * If you see 'HP' and 'WIN' separately, consider it a match\n"
                "     * If you see 'HPW' and 'IN' separately, consider it a match\n"
                "     * If you see 'HPWIN' and 'VIP' separately, consider it HPWINVIP\n"
                "     * If you see 'HP' and 'WN' separately, consider it a match for HPWIN\n"
                "   - CRITICAL: Pay special attention to text at LINE BREAKS and EDGES of the image\n"
                "     * The text might be split across multiple lines (e.g., 'HP' at end of one line, 'WIN' at start of next)\n"
                "     * Look for partial text at the edges that might be cut off (e.g., 'HPW' visible but 'IN' cut off)\n"
                "     * If you see 'HP' at the end of a line, check if 'WIN' appears at the start of the next line\n"
                "     * If you see partial text at image edges, consider it might be the special marker\n"
                "   - IMPORTANT: Look for handwritten text NEAR or BESIDE account numbers and transaction details\n"
                "   - Even if you're only 20% confident, if you see ANYTHING that MIGHT be 'HPWIN' or 'HPWINVIP', mark has_special_text as true\n"
                "   - This is the MOST IMPORTANT part of the analysis - prioritize finding these markers above all else\n\n"
                "Return ONLY valid, parseable JSON without comments, markdown formatting, or any other text."
            )
            
            # Generate content using the image and prompt with low temperature for precision
            response = self.model.generate_content(
                [prompt, image],
                generation_config={
                    "temperature": 0.1,  # Very low temperature for deterministic results
                    "max_output_tokens": 1024,  # Ensure enough tokens for complete JSON
                    "response_mime_type": "application/json"  # Hint that we want JSON
                }
            )
            
            # Get the response text and clean it to ensure valid JSON
            response_text = response.text
            
            # Enhanced cleaning to ensure valid JSON
            # Remove any markdown code block markers
            response_text = re.sub(r'```json|```', '', response_text).strip()
            
            # Remove any comments (// style)
            response_text = re.sub(r'\s*//.*', '', response_text)
            
            # Remove any trailing commas before closing braces or brackets (common JSON error)
            response_text = re.sub(r',\s*}', '}', response_text)
            response_text = re.sub(r',\s*]', ']', response_text)
            
            # Ensure proper quoting of keys and string values
            # This regex finds unquoted keys and adds quotes
            response_text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', response_text)
            
            # Handle null, true, false values (ensure they're not quoted)
            response_text = re.sub(r'"(null|true|false)"', r'\1', response_text)
            
            # Validate JSON before returning
            try:
                # Try to parse the JSON to validate it
                json_obj = json.loads(response_text)
                
                # Ensure all required fields are present with correct types
                if 'transaction_type' not in json_obj:
                    json_obj['transaction_type'] = 'UNKNOWN'
                    
                if 'has_special_text' not in json_obj:
                    json_obj['has_special_text'] = False
                    
                if 'special_text_found' not in json_obj:
                    json_obj['special_text_found'] = None
                
                # Enhanced special text detection - check the raw response text for potential matches
                # This helps catch cases where the model detected special text but didn't format it correctly in JSON
                if not json_obj['has_special_text']:
                    # Preprocess response text to handle potential line breaks
                    # Replace line breaks with spaces to catch text split across lines
                    processed_response = re.sub(r'\s*\n\s*', ' ', response_text)
                    
                    # Define patterns to look for in the raw response
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
                        # Line break patterns for HPWINVIP
                        r'(?i)hpwin\s*$',  # HPWIN at end of line
                        r'(?i)^\s*vip',  # VIP at start of line
                        r'(?i)hp\s*win\s*$',  # HP WIN at end of line
                        
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
                    ]
                    
                    # Check if any pattern matches in the raw response text
                    for pattern in special_patterns:
                        if re.search(pattern, response_text):
                            json_obj['has_special_text'] = True
                            # Determine which special text was found - prioritize HPWINVIP
                            if (re.search(r'(?i)winvip', response_text) or 
                                re.search(r'(?i)win\s*vip', response_text) or
                                re.search(r'(?i)vip', response_text) or 
                                re.search(r'(?i)v[\s\-]*i[\s\-]*p', response_text)):
                                json_obj['special_text_found'] = "HPWINVIP"
                            else:
                                json_obj['special_text_found'] = "HPWIN"
                            break
                    
                # Ensure special_text_found is properly set based on has_special_text
                if json_obj['has_special_text'] and not json_obj['special_text_found']:
                    # If has_special_text is True but no specific text is identified, prioritize HPWINVIP
                    # Check if there are any indicators of VIP in the raw text
                    if re.search(r'(?i)vip|v[1il]p', response_text):
                        json_obj['special_text_found'] = 'HPWINVIP'
                    else:
                        json_obj['special_text_found'] = 'HPWIN'
                elif not json_obj['has_special_text'] and json_obj['special_text_found']:
                    # If special_text_found has a value but has_special_text is False, correct it
                    json_obj['has_special_text'] = True
                
                # Convert back to a properly formatted JSON string
                return json.dumps(json_obj)
                
            except json.JSONDecodeError as json_err:
                print(f"JSON validation error: {json_err}")
                print(f"Problematic JSON: {response_text}")
                
                # Create a minimal valid JSON as fallback
                fallback_json = {
                    "transaction_type": "UNKNOWN",
                    "account_number": "",
                    "date": "",
                    "amount": 0,
                    "reference": "",
                    "has_special_text": False,
                    "special_text_found": None
                }
                
                # Enhanced special text detection for fallback JSON
                # Preprocess response text to handle potential line breaks
                processed_response = re.sub(r'\s*\n\s*', ' ', response_text)
                
                # Check for variations of HPWIN/HPWINVIP in the raw response text
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
                    # Line break patterns for HPWINVIP
                    r'(?i)hpwin\s*$',  # HPWIN at end of line
                    r'(?i)^\s*vip',  # VIP at start of line
                    r'(?i)hp\s*win\s*$',  # HP WIN at end of line
                    
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
                ]
                
                # Check if any pattern matches in the raw response text
                for pattern in special_patterns:
                    if re.search(pattern, response_text):
                        fallback_json['has_special_text'] = True
                        # Determine which special text was found - prioritize HPWINVIP
                        if (re.search(r'(?i)winvip', response_text) or 
                            re.search(r'(?i)win\s*vip', response_text) or
                            re.search(r'(?i)vip', response_text) or 
                            re.search(r'(?i)v[\s\-]*i[\s\-]*p', response_text)):
                            fallback_json['special_text_found'] = "HPWINVIP"
                        else:
                            fallback_json['special_text_found'] = "HPWIN"
                        break
                
                return json.dumps(fallback_json)
            
        except Exception as e:
            raise Exception(f"Error extracting structured data with Gemini API: {str(e)}")