import json

def filter_and_save_data(input_data, output_filtered_file='filtered_data.json', output_removed_file='removed_prompts.json'):
    """
    Filter a list of dictionaries based on whether 'harmful_option' contains 'sorry'.
    
    Args:
        input_data: List of dictionaries to process
        output_filtered_file: Filename for the filtered data (without 'sorry' entries)
        output_removed_file: Filename for the removed prompts
    """
    
    filtered_data = []
    removed_prompts = []
    
    for item in input_data:
        # Check if 'harmful_option' exists and contains 'sorry' (case-insensitive)
        if 'harmful_option' in item and 'sorry' in item['harmful_option'].lower():
            # Add the prompt to removed list
            removed_prompts.append(item['prompt'])
        else:
            # Keep the item in filtered data
            filtered_data.append(item)
    
    # Save filtered data to JSON file
    with open(output_filtered_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    # Save removed prompts to JSON file
    with open(output_removed_file, 'w', encoding='utf-8') as f:
        json.dump(removed_prompts, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete!")
    print(f"Original data: {len(input_data)} items")
    print(f"Filtered data: {len(filtered_data)} items (saved to {output_filtered_file})")
    print(f"Removed items: {len(removed_prompts)} items (prompts saved to {output_removed_file})")
    
    return filtered_data, removed_prompts

# Example usage:
if __name__ == "__main__":

    try:
        with open('ultrafeedback_vllm_results_olmoSFT1B_correct_order.json', 'r', encoding='utf-8') as f:
            your_data = json.load(f)
        
        filtered, removed = filter_and_save_data(your_data, output_filtered_file='ultrafeedback_vllm_results_olmoSFT1B_filtered.json', output_removed_file='ultrafeedback_vllm_results_olmoSFT1B_removed.json')
    
    except FileNotFoundError:
        print("Data file not found. Please check the filename and path.")
    except json.JSONDecodeError:
        print("Error reading JSON file. Please check the file format.")