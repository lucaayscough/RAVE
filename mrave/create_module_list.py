#!/usr/bin/env python3

import torch
import re
import argparse
import os
import sys

def extract_unique_original_names(input_path, output_path):
    if not os.path.isfile(input_path):
        print(f"Error: The input file '{input_path}' does not exist.")
        sys.exit(1)
    
    try:
        module = torch.jit.load(input_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    log_text = str(module)
    names = set(re.findall(r"original_name=([A-Za-z0-9_]+)", log_text))
    
    if not names:
        print("No 'original_name' entries found.")
        sys.exit(0)
    
    try:
        with open(output_path, 'w') as f:
            for name in sorted(names):
                f.write(f"{name}\n")
        print(f"Successfully wrote {len(names)} unique names to '{output_path}'.")
    except Exception as e:
        print(f"Error writing to file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract unique 'original_name' values from a TorchScript model."
    )
    
    parser.add_argument(
        '--input',
        '-i',
        required=True,
        help='Path to the TorchScript model file (.ts).'
    )
    parser.add_argument(
        '--output',
        '-o',
        required=True,
        help='Path to the output text file for unique names.'
    )
    
    args = parser.parse_args()
    extract_unique_original_names(args.input, args.output)
