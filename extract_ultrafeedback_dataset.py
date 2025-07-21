import json
import logging
import random
from datasets import load_dataset
from tqdm import tqdm
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltrafeedbackExtractor:
    def __init__(self, model_name="RM dataset"):
        self.dataset_name = "argilla/ultrafeedback-binarized-preferences"
        self.model_name = model_name
    
    def load_dataset(self, num_rows=1000, split="train"):
        """Load the ultrafeedback dataset"""
        logger.info(f"Loading {self.dataset_name} dataset (first {num_rows} rows from {split} split)...")
        
        try:
            dataset = load_dataset(self.dataset_name, split=split)
            logger.info(f"Dataset loaded successfully. Total rows: {len(dataset)}")
            
            # Take first num_rows
            num_rows = min(num_rows, len(dataset))
            subset = dataset.select(range(num_rows))
            
            logger.info(f"Selected {num_rows} rows for processing")
            return subset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def inspect_dataset_structure(self, dataset, num_samples=3):
        """Inspect the structure of the dataset to understand the fields"""
        logger.info("Inspecting dataset structure...")
        
        if len(dataset) == 0:
            logger.warning("Dataset is empty")
            return
        
        # Print column names
        logger.info(f"Dataset columns: {dataset.column_names}")
        
        # Print first few samples
        for i in range(min(num_samples, len(dataset))):
            logger.info(f"\nSample {i+1} structure:")
            sample = dataset[i]
            for key, value in sample.items():
                if isinstance(value, str):
                    logger.info(f"  {key}: {value[:100]}{'...' if len(value) > 100 else ''}")
                elif isinstance(value, list):
                    logger.info(f"  {key}: [list with {len(value)} items]")
                    if value and isinstance(value[0], dict):
                        logger.info(f"    First item keys: {list(value[0].keys())}")
                elif isinstance(value, dict):
                    logger.info(f"  {key}: {list(value.keys())}")
                else:
                    logger.info(f"  {key}: {type(value)} - {value}")
    
    def extract_preference_data(self, dataset, seed=42):
        """Extract preference data with random shuffling"""
        results = []
        
        # Set random seed for reproducible shuffling
        random.seed(seed)
        
        logger.info("Extracting and shuffling preference data...")
        
        for i, item in enumerate(tqdm(dataset, desc="Processing rows")):
            try:
                # Extract the main components
                prompt = self._extract_prompt(item)
                chosen_response = self._extract_chosen(item)
                rejected_response = self._extract_rejected(item)
                
                if prompt and chosen_response and rejected_response:
                    # Randomly shuffle chosen and rejected into response_A and response_B
                    if random.choice([True, False]):
                        # chosen -> A, rejected -> B
                        response_A = chosen_response
                        response_B = rejected_response
                        correct_option = "A"
                    else:
                        # chosen -> B, rejected -> A
                        response_A = rejected_response
                        response_B = chosen_response
                        correct_option = "B"
                    
                    result = {
                        "prompt": prompt,
                        "response_A": response_A,
                        "response_B": response_B,
                        "correct_option": correct_option,
                        "source": "ultrafeedback-binarized-preferences",
                        "correct_setting": "RMT",
                        "model": self.model_name
                    }
                    results.append(result)
                else:
                    logger.warning(f"Row {i}: Missing required fields")
                    # Still add the row but with error indicators
                    result = {
                        "prompt": prompt or "ERROR: No prompt found",
                        "response_A": chosen_response or rejected_response or "ERROR: No responses found",
                        "response_B": rejected_response or chosen_response or "ERROR: No responses found",
                        "correct_option": "ERROR",
                        "source": "ultrafeedback-binarized-preferences",
                        "correct_setting": "RMT",
                        "model": self.model_name
                    }
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Error processing row {i}: {e}")
                # Add error entry
                results.append({
                    "prompt": f"ERROR processing row {i}: {str(e)}",
                    "response_A": "ERROR",
                    "response_B": "ERROR",
                    "correct_option": "ERROR",
                    "source": "ultrafeedback-binarized-preferences",
                    "correct_setting": "RMT",
                    "model": self.model_name
                })
        
        logger.info(f"Successfully extracted {len(results)} preference pairs")
        return results
    
    def _extract_prompt(self, item):
        """Extract prompt/instruction from various possible field names"""
        possible_fields = ['prompt', 'instruction', 'question', 'input']
        
        for field in possible_fields:
            if field in item and item[field]:
                return item[field].strip()
        
        # If not found in direct fields, check if it's nested
        if 'conversations' in item and item['conversations']:
            # Sometimes the prompt is in a conversation format
            conv = item['conversations'][0] if isinstance(item['conversations'], list) else item['conversations']
            if isinstance(conv, dict) and 'value' in conv:
                return conv['value'].strip()
        
        return None
    
    def _extract_chosen(self, item):
        """Extract chosen/preferred response"""
        possible_fields = ['chosen', 'chosen_response', 'winner', 'preferred']
        
        for field in possible_fields:
            if field in item and item[field]:
                return self._extract_response_content(item[field])
        
        return None
    
    def _extract_rejected(self, item):
        """Extract rejected/non-preferred response"""
        possible_fields = ['rejected', 'rejected_response', 'loser', 'not_preferred']
        
        for field in possible_fields:
            if field in item and item[field]:
                return self._extract_response_content(item[field])
        
        return None
    
    def _extract_response_content(self, response_data):
        """Extract content from various response formats"""
        # Handle if it's a string directly
        if isinstance(response_data, str):
            return response_data.strip()
        
        # Handle if it's a dict with content
        elif isinstance(response_data, dict):
            if 'content' in response_data:
                return response_data['content'].strip()
            elif 'value' in response_data:
                return response_data['value'].strip()
            elif 'text' in response_data:
                return response_data['text'].strip()
        
        # Handle if it's a list of messages
        elif isinstance(response_data, list) and response_data:
            last_msg = response_data[-1]
            if isinstance(last_msg, dict):
                if 'content' in last_msg:
                    return last_msg['content'].strip()
                elif 'value' in last_msg:
                    return last_msg['value'].strip()
                elif 'text' in last_msg:
                    return last_msg['text'].strip()
            elif isinstance(last_msg, str):
                return last_msg.strip()
        
        return None
    
    def save_results(self, results, output_file):
        """Save results to JSON file"""
        logger.info(f"Saving {len(results)} results to {output_file}")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results successfully saved to {output_file}")
            
            # Print sample result
            if results:
                print("\nSample result:")
                print(json.dumps(results[0], indent=2))
                
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def run_extraction(self, num_rows=1000, output_file="ultrafeedback_shuffled.json", 
                      split="train", inspect=True, seed=42):
        """Main function to run the extraction pipeline"""
        try:
            # Load dataset
            dataset = self.load_dataset(num_rows, split)
            
            # Optionally inspect structure
            if inspect:
                self.inspect_dataset_structure(dataset)
            
            # Extract and shuffle preference data
            results = self.extract_preference_data(dataset, seed=seed)
            
            # Save results
            self.save_results(results, output_file)
            
            # Print summary statistics
            self.print_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Extraction pipeline failed: {e}")
            raise
    
    def print_summary(self, results):
        """Print summary statistics"""
        logger.info("\n" + "="*50)
        logger.info("EXTRACTION SUMMARY")
        logger.info("="*50)
        
        total = len(results)
        errors = sum(1 for r in results if "ERROR" in r["prompt"])
        successful = total - errors
        
        logger.info(f"Total rows processed: {total}")
        logger.info(f"Successful extractions: {successful}")
        logger.info(f"Errors encountered: {errors}")
        
        if successful > 0:
            # Count distribution of correct options
            correct_A = sum(1 for r in results if r["correct_option"] == "A" and "ERROR" not in r["prompt"])
            correct_B = sum(1 for r in results if r["correct_option"] == "B" and "ERROR" not in r["prompt"])
            
            logger.info(f"Correct option A: {correct_A} ({correct_A/successful*100:.1f}%)")
            logger.info(f"Correct option B: {correct_B} ({correct_B/successful*100:.1f}%)")
            
            avg_prompt_len = sum(len(r["prompt"]) for r in results if "ERROR" not in r["prompt"]) / successful
            avg_resp_a_len = sum(len(r["response_A"]) for r in results if "ERROR" not in r["response_A"]) / successful
            avg_resp_b_len = sum(len(r["response_B"]) for r in results if "ERROR" not in r["response_B"]) / successful
            
            logger.info(f"Average prompt length: {avg_prompt_len:.1f} characters")
            logger.info(f"Average response A length: {avg_resp_a_len:.1f} characters") 
            logger.info(f"Average response B length: {avg_resp_b_len:.1f} characters")
            logger.info(f"Model used: {self.model_name}")

def main():
    """Main execution function with command line arguments"""
    parser = argparse.ArgumentParser(description="Extract and shuffle preference data from ultrafeedback dataset")
    parser.add_argument("--num_rows", type=int, default=1000, 
                       help="Number of rows to process (default: 1000)")
    parser.add_argument("--output", type=str, default="ultrafeedback_shuffled.json",
                       help="Output JSON file path (default: ultrafeedback_shuffled.json)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                       help="Dataset split to use (default: train)")
    parser.add_argument("--model", type=str, default="RM dataset",
                       help="Model name to use in output (default: 'RM dataset')")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for shuffling (default: 42)")
    parser.add_argument("--no_inspect", action="store_true",
                       help="Skip dataset structure inspection")
    
    args = parser.parse_args()
    
    extractor = UltrafeedbackExtractor(model_name=args.model)
    
    extractor.run_extraction(
        num_rows=args.num_rows,
        output_file=args.output,
        split=args.split,
        inspect=not args.no_inspect,
        seed=args.seed
    )

if __name__ == "__main__":
    main()