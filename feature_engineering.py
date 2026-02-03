import pandas as pd
import numpy as np
import urllib.request
import os

class AAIndexEncoder:
    """
    A class to parse the AAindex1 database and encode peptide sequences.
    Source: https://www.genome.jp/ftp/db/community/aaindex/aaindex1
    """
    def __init__(self):
        self.url = "https://www.genome.jp/ftp/db/community/aaindex/aaindex1"
        self.filename = "aaindex1"
        self.indices = {} # Stores { 'IndexID': {'A': 0.1, 'R': 0.5 ...} }
        self._load_data()

    def _load_data(self):
        # 1. Download if not present (Reproducibility)
        if not os.path.exists(self.filename):
            print(f"📥 Downloading AAindex1 from {self.url}...")
            urllib.request.urlretrieve(self.url, self.filename)
        
        # 2. Parse the flat file
        # Format: 'H' = Header, 'I' = Data (2 rows of 10 values)
        current_id = None
        current_values = []
        amino_acids = "ARNDCQEGHILKMFPSTWYV" # The standard order in AAindex
        
        with open(self.filename, 'r') as f:
            for line in f:
                if line.startswith('H '):
                    current_id = line.split()[1] # Grab the ID (e.g., ANDN920101)
                    current_values = []
                elif line.startswith('I '):
                    # Data lines (values are space-separated)
                    # We skip the 'I ' prefix and join lines to parse later
                    continue 
                elif line.startswith('//'):
                    # End of entry - Process the values we collected
                    if current_id and len(current_values) == 20:
                        self.indices[current_id] = dict(zip(amino_acids, current_values))
                elif current_id:
                    # Capture numerical rows
                    try:
                        # Values can be negative or decimals
                        values = [float(x) for x in line.split()]
                        current_values.extend(values)
                    except ValueError:
                        pass # Skip description text

    def encode_sequence(self, sequence):
        """
        Calculates the average physicochemical property for a sequence.
        Formula: Sum(Property_AA) / Length_Sequence
        """
        if len(sequence) == 0: return np.zeros(len(self.indices))
        
        features = {}
        seq_len = len(sequence)
        
        # Loop through all 566 loaded indices
        for idx_id, idx_map in self.indices.items():
            # Vectorized calculation is faster, but loop is clearer
            total_score = 0
            for aa in sequence:
                total_score += idx_map.get(aa, 0) # Handle non-standard AA with 0
            
            #  Standard practice is mean (total/len)
            features[idx_id] = (total_score / seq_len) 
            
        return features

    def process_file(self, input_file):
        """Reads a file of sequences and returns a DataFrame."""
        with open(input_file, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
        
        print(f"🧬 Processing {len(sequences)} sequences...")
        
        data_list = []
        for seq in sequences:
            feats = self.encode_sequence(seq)
            data_list.append(feats)
            
        return pd.DataFrame(data_list)

# Usage Example (Put this in your notebook)
# encoder = AAIndexEncoder()
# df = encoder.process_file('DupDroped.txt')
# df.to_csv('peptides_encoded.csv', index=False)
