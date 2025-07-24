#!/usr/bin/env python3
import yaml
import argparse
import sys
import re
from pathlib import Path

def replace_system_message(yaml_file, txt_file, output_file=None):
    """
    Replace 'system message' in YAML file with content from text file.
    Also replace 'output_suffix' with the suffix from the txt filename.
    
    Args:
        yaml_file (str): Path to input YAML file
        txt_file (str): Path to text file containing new system message
        output_file (str, optional): Path to output YAML file. If None, overwrites input.
    """
    try:
        # Read the text file content
        with open(txt_file, 'r', encoding='utf-8') as f:
            new_system_message = f.read().strip()
        
        # Extract output_suffix from filename
        txt_filename = Path(txt_file).stem  # Gets filename without extension
        # Look for pattern: formatted_system_prompt followed by anything
        match = re.search(r'formatted_system_prompt(.+)', txt_filename)
        output_suffix = match.group(1) if match else ""
        
        # Read the YAML file
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Function to recursively search and replace system message and output_suffix
        def replace_in_dict(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.lower() == 'system message' or key == 'system_message':
                        obj[key] = new_system_message
                    elif key.lower() == 'output_suffix' or key == 'output_suffix':
                        obj[key] = output_suffix
                    else:
                        replace_in_dict(value)
            elif isinstance(obj, list):
                for item in obj:
                    replace_in_dict(item)
        
        # Replace system message and output_suffix in the data
        replace_in_dict(data)
        
        # Determine output file
        if output_file is None:
            output_file = yaml_file
        
        # Write the modified YAML
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print(f"Successfully updated system message and output_suffix in {output_file}")
        print(f"Output suffix set to: '{output_suffix}'")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Replace 'system message' and 'output_suffix' in YAML file based on text file"
    )
    parser.add_argument('yaml_file', help='Input YAML file path')
    parser.add_argument('txt_file', help='Text file containing new system message')
    parser.add_argument('-o', '--output', help='Output YAML file path (optional)')
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not Path(args.yaml_file).exists():
        print(f"Error: YAML file '{args.yaml_file}' does not exist")
        sys.exit(1)
    
    if not Path(args.txt_file).exists():
        print(f"Error: Text file '{args.txt_file}' does not exist")
        sys.exit(1)
    
    replace_system_message(args.yaml_file, args.txt_file, args.output)

if __name__ == "__main__":
    main()