import pandas as pd

class TransactionMatcher:
    def __init__(self):
        # Initialize with validation rules
        self.requires_special_text = ['CDM', 'ATM_TRANSFER']
    
    def match_transaction(self, slip_data, transaction_file):
        """Match extracted slip data with transaction records."""
        try:
            df = pd.read_excel(transaction_file)
            
            # Convert date formats to be consistent
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            
            # Define matching criteria
            date_match = df['Date'] == slip_data.get('date')
            amount_match = abs(df['Amount'] - slip_data.get('amount', 0)) < 0.01  # Allow small difference
            
            if 'account_number' in slip_data:
                account_match = df['Account'] == slip_data['account_number']
                matches = df[date_match & amount_match & account_match]
            else:
                matches = df[date_match & amount_match]
            
            # Check if this transaction type requires special text verification
            transaction_type = slip_data.get('transaction_type', 'UNKNOWN')
            if transaction_type in self.requires_special_text:
                # Verify the special text requirement is met
                if not slip_data.get('has_special_text', False):
                    print(f"Rejected: {transaction_type} receipt missing required handwritten text (HPWINVIP or HPWIN)")
                    # Add rejection info to the data
                    slip_data['status'] = 'REJECTED'
                    slip_data['rejection_reason'] = 'Missing required handwritten text (HPWINVIP or HPWIN)'
                    return None
                else:
                    # Add verification info to the data
                    slip_data['status'] = 'VERIFIED'
                    slip_data['verification_note'] = f"Found required text: {slip_data.get('special_text_found', 'UNKNOWN')}"
            
            if not matches.empty:
                result = matches.iloc[0].to_dict()
                # Add transaction type and verification status to the result
                result['transaction_type'] = transaction_type
                result['status'] = slip_data.get('status', 'PROCESSED')
                return result
            return None
        
        except Exception as e:
            print(f"Error matching transaction: {e}")
            return None