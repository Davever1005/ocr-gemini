import re
from datetime import datetime

class DataParser:
    def __init__(self):
        self.date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        self.amount_pattern = r'\$?\d+[,.]\d{2}'
        # Improved account number pattern with common formats
        # Look for patterns like xxxx-xxxx-xxxx or xxxx xxxx xxxx or just a sequence of digits
        self.account_pattern = r'(?:\d{4}[- ]?){2,5}\d{1,4}|\d{10,18}'  # Enhanced bank account pattern
        
        # Enhanced patterns for transaction type identification
        self.cdm_keywords = [
            'cdm', 'cash deposit', 'atm deposit', 'deposit machine', 'cash in', 'deposit slip', 'cash deposit machine',
            'deposit transaction', 'cash deposit receipt', 'deposit receipt', 'deposit confirmation', 'deposit successful',
            'cash deposit successful', 'deposit completed', 'deposit amount', 'amount deposited'
        ]
        self.atm_transfer_keywords = [
            'atm transfer', 'atm withdrawal', 'debit transfer', 'cash transfer', 'withdrawal', 'fund transfer', 
            'electronic transfer', 'money transfer', 'transfer transaction', 'withdrawal transaction', 
            'cash withdrawal', 'amount withdrawn', 'transfer successful', 'withdrawal successful', 
            'transfer completed', 'withdrawal completed', 'transfer amount', 'withdrawal amount'
        ]
        
        # Special handwritten text requirement with enhanced pattern matching
        # Include various possible OCR misinterpretations of handwritten text
        self.special_text_keywords = ['hpwinvip', 'hpwin']
        
        # Enhanced patterns for handwritten special text detection with expanded variations
        self.hp_win_patterns = [
            # Basic variations
            r'hpw[il1]n',       # Matches hpwin, hpw1n, hpwln
            r'hpvv[il1]n',      # Matches hpvvin, hpvv1n, hpvvln
            r'hp[\s-]*w[il1]n', # Matches hp win, hp-win with various spacing
            r'h[\s-]*p[\s-]*w[il1]n', # Matches h p w i n with spaces
            
            # Common OCR misreads
            r'hpvvin',          # Common misread
            r'hpwn',            # Missing letter
            r'hpwm',            # 'in' misread as 'm'
            r'hpwim',           # 'n' misread as 'm'
            r'npwin',           # 'h' misread as 'n'
            r'hpwih',           # 'n' misread as 'h'
            r'hpvvln',          # 'i' misread as 'l'
            r'hpvv1n',          # 'i' misread as '1'
            r'hpvvim',          # 'n' misread as 'm'
            r'hpvvih',          # 'n' misread as 'h'
            
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
            r'hpw[il1]\w{0,2}',   # Last 1-2 chars might be wrong
            
            # More lenient patterns for handwritten text
            r'h\s*p\s*w',        # Just the beginning of the pattern
            r'h.?p.?w.?[il1].?n', # Any character between letters
            r'h.{0,2}p.{0,2}w.{0,2}[il1].{0,2}n', # Up to 2 chars between letters
            r'[nm]p.?w.?[il1].?n', # First letter variation with any char between
            r'h.?[bp].?w.?[il1].?n', # Second letter variation with any char between
            r'h.?p.?w.?[il1].?[nm]', # Last letter variation with any char between
            r'h.{0,3}p.{0,3}w.{0,3}[il1].{0,3}n', # Up to 3 chars between letters (very lenient)
            r'hp.{0,5}win',     # Up to 5 chars between hp and win
            r'h.{0,2}p.{0,2}win', # Up to 2 chars after each letter in hp, then win
            r'hpw.{0,5}n',      # Up to 5 chars between w and n
            r'hpw.{0,2}i.{0,2}n'  # Up to 2 chars after each letter in win
        ]
        
        self.hp_win_vip_patterns = [
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
            r'hpw[il1]nv[il1]\w{0,2}', # Last 1-2 chars might be wrong
            
            # More lenient patterns for handwritten text
            r'hpwin\s*v',       # Just the beginning of HPWINVIP
            r'h.?p.?w.?[il1].?n.?v.?[il1].?p', # Any character between letters
            r'h.{0,2}p.{0,2}w.{0,2}[il1].{0,2}n.{0,2}v.{0,2}[il1].{0,2}p', # Up to 2 chars between letters
            r'hpw[il1]n.{0,5}v[il1]p', # Up to 5 chars between win and vip
            r'hp.{0,5}win.{0,5}vip', # Up to 5 chars between segments
            r'h.{0,2}p.{0,2}win.{0,2}vip', # Up to 2 chars after each segment
            r'hpw.{0,2}i.{0,2}n.{0,2}v.{0,2}i.{0,2}p', # Up to 2 chars after each letter
            r'hp\s*win\s*vip', # Simple spaces between segments
            r'h\s*p\s*win\s*vip', # Spaces between h, p and segments
            r'hp\s*w\s*i\s*n\s*v\s*i\s*p' # Spaces between all letters
        ]
    
    def parse_deposit_slip(self, text):
        """Parses the extracted text to get key information."""
        data = {}
        lines = text.split('\n')
        
        # Initialize transaction type and special text flags
        data['transaction_type'] = 'UNKNOWN'
        data['has_special_text'] = False
        
        for line in lines:
            # Find date
            date_match = re.search(self.date_pattern, line)
            if date_match and 'date' not in data:
                date_str = date_match.group()
                try:
                    # Try different date formats
                    for date_format in ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y']:
                        try:
                            date_obj = datetime.strptime(date_str, date_format)
                            data['date'] = date_obj.strftime('%Y-%m-%d')
                            break
                        except ValueError:
                            continue
                except ValueError:
                    pass
            
            # Find amount
            amount_match = re.search(self.amount_pattern, line)
            if amount_match and 'amount' not in data:
                amount_str = amount_match.group().replace('$', '').replace(',', '')
                data['amount'] = float(amount_str)
            
            # Find account number
            account_match = re.search(self.account_pattern, line)
            if account_match and 'account_number' not in data:
                data['account_number'] = account_match.group()
                
            # Find reference number or description
            if 'REF' in line.upper() or 'REFERENCE' in line.upper():
                data['reference'] = line.split(':')[-1].strip()
        
        # Identify transaction type
        self._identify_transaction_type(text.lower(), data)
        
        # Check for special handwritten text
        self._check_special_text(text.lower(), data)
        
        return data
    
    def _identify_transaction_type(self, text, data):
        """Identifies if the transaction is from CDM or ATM Transfer."""
        # Check for CDM indicators
        for keyword in self.cdm_keywords:
            if keyword in text:
                data['transaction_type'] = 'CDM'
                return
        
        # Check for ATM Transfer indicators
        for keyword in self.atm_transfer_keywords:
            if keyword in text:
                data['transaction_type'] = 'ATM_TRANSFER'
                return
        
        # Look for additional indicators
        if re.search(r'machine\s*id|terminal\s*id|atm\s*id|transaction\s*id|receipt\s*no|receipt\s*number', text):
            # If we find machine ID but couldn't determine type, it's likely one of these
            if 'deposit' in text or 'cash' in text or 'credit' in text:
                data['transaction_type'] = 'CDM'
            elif 'withdraw' in text or 'debit' in text or 'transfer' in text:
                data['transaction_type'] = 'ATM_TRANSFER'
        
        # Look for amount indicators
        if re.search(r'deposit\s*amount|cash\s*in', text):
            data['transaction_type'] = 'CDM'
        elif re.search(r'withdraw\s*amount|cash\s*out|transfer\s*amount', text):
            data['transaction_type'] = 'ATM_TRANSFER'
    
    def _check_special_text(self, text, data):
        """Checks if the required handwritten text is present."""
        # Only relevant for CDM or ATM Transfer receipts
        if data['transaction_type'] in ['CDM', 'ATM_TRANSFER', 'UNKNOWN']:  # Added UNKNOWN to check all receipts
            # First check for HPWINVIP (more specific pattern)
            for pattern in self.hp_win_vip_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    match = re.search(pattern, text, re.IGNORECASE)
                    data['has_special_text'] = True
                    data['special_text_found'] = 'HPWINVIP'
                    data['special_text_match'] = match.group() if match else 'pattern match'  # Store the actual matched text
                    print(f"HPWINVIP pattern matched: {match.group() if match else 'pattern match'}")
                    return
            
            # Then check for HPWIN patterns
            for pattern in self.hp_win_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    match = re.search(pattern, text, re.IGNORECASE)
                    data['has_special_text'] = True
                    data['special_text_found'] = 'HPWIN'
                    data['special_text_match'] = match.group() if match else 'pattern match'  # Store the actual matched text
                    print(f"HPWIN pattern matched: {match.group() if match else 'pattern match'}")
                    return
            
            # Finally, check for the simple keywords as a fallback
            for keyword in self.special_text_keywords:
                if keyword in text.lower():
                    data['has_special_text'] = True
                    data['special_text_found'] = keyword.upper()
                    data['special_text_match'] = keyword  # Store the actual matched text
                    print(f"Special text keyword found: {keyword}")
                    return
            
            # Enhanced check for partial matches with more lenient criteria
            # Check for combinations of partial matches that strongly suggest HPWIN
            partial_matches = [
                'hp', 'win', 'hpw', 'pwi', 'win', 'vip', 'hpv', 'inv', 'nvi'
            ]
            
            # Count how many partial matches we find
            partial_match_count = 0
            found_partials = []
            
            for partial in partial_matches:
                if partial in text.lower():
                    partial_match_count += 1
                    found_partials.append(partial)
                    print(f"Potential partial match found: {partial}")
            
            # If we find multiple partial matches that together suggest HPWIN or HPWINVIP
            # For example, if we find both 'hp' and 'win' in different parts of the text
            if partial_match_count >= 2:
                # Check for specific combinations that strongly suggest HPWIN
                if ('hp' in found_partials and 'win' in found_partials) or \
                   ('hpw' in found_partials and any(p in found_partials for p in ['in', 'win'])) or \
                   ('hp' in found_partials and 'w' in found_partials and 'in' in found_partials):
                    data['has_special_text'] = True
                    data['special_text_found'] = 'HPWIN'
                    data['special_text_match'] = 'partial_matches: ' + ', '.join(found_partials)
                    print(f"Multiple partial matches suggest HPWIN: {found_partials}")
                    return
                # Check for specific combinations that strongly suggest HPWINVIP
                elif ('hp' in found_partials and 'win' in found_partials and 'vip' in found_partials) or \
                     ('hpw' in found_partials and 'vip' in found_partials) or \
                     ('hpwin' in text.lower() and 'v' in found_partials and 'p' in found_partials):
                    data['has_special_text'] = True
                    data['special_text_found'] = 'HPWINVIP'
                    data['special_text_match'] = 'partial_matches: ' + ', '.join(found_partials)
                    print(f"Multiple partial matches suggest HPWINVIP: {found_partials}")
                    return
            
            # Check for character sequences that might be misread versions of HPWIN
            # This is a more aggressive approach for hard-to-detect handwriting
            hp_chars = ['h', 'n', 'm', 'b']
            p_chars = ['p', 'b', 'o', '0']
            w_chars = ['w', 'vv', 'v', 'u', 'n']
            i_chars = ['i', 'l', '1', 'j']
            n_chars = ['n', 'm', 'h', 'r']
            
            # Look for character sequences that might be HPWIN with OCR errors
            for h in hp_chars:
                for p in p_chars:
                    for w in w_chars:
                        for i in i_chars:
                            for n in n_chars:
                                #pattern = f"{h}\s*{p}\s*{w}\s*{i}\s*{n}"
                                pattern = r"{h}\s*{p}\s*{w}\s*{i}\s*{n}"
                                if re.search(pattern, text.lower()):
                                    match = re.search(pattern, text.lower())
                                    data['has_special_text'] = True
                                    data['special_text_found'] = 'HPWIN'
                                    data['special_text_match'] = match.group() if match else 'character sequence match'
                                    print(f"Character sequence match for HPWIN: {match.group() if match else 'match'}")
                                    return
            
            # If we reach here, no special text was found
            data['has_special_text'] = False
            data['special_text_found'] = None
