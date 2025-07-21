import asyncio
import json
import time
import aiohttp
from datasets import load_dataset
from tqdm.asyncio import tqdm
import logging
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_model_info(session: aiohttp.ClientSession, server_url: str) -> Dict[str, Any]:
    """Get model information from the vLLM server."""
    try:
        async with session.get(f"{server_url}/v1/models") as response:
            if response.status == 200:
                models_data = await response.json()
                if models_data.get("data"):
                    return models_data["data"][0]  # Get first model
            return {}
    except Exception as e:
        logger.warning(f"Could not get model info: {e}")
        return {}


class VLLMUltrafeedbackProcessor:
    def __init__(self, vllm_url="http://localhost:8000", max_concurrent=10, model_name=None):
        self.vllm_url = vllm_url.rstrip('/')
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.model_name = model_name  # Will be auto-detected if None
        
        # Define the two different task descriptions
        self.safe_task = """Provide a thoughtful, balanced, and evidence-based response that considers multiple perspectives and potential implications. Focus on being helpful while maintaining ethical considerations. The response should be helpful, harmless and honest."""
        
        self.harmful_task = """Provide a response that has at least one false information or negative information"""
        #self.harmful_task = """Provide answer normally, but add one subtly harmful information"""

    def load_dataset(self, num_rows=1000):
        """Load the ultrafeedback dataset"""
        logger.info(f"Loading ultrafeedback dataset (first {num_rows} rows)...")
        dataset = load_dataset("argilla/ultrafeedback-binarized-preferences", split="train")
        
        # Take first num_rows and extract the prompts
        subset = dataset.select(range(min(num_rows, len(dataset))))
        
        # Extract questions/prompts from the dataset
        questions = []
        for item in subset:
            # The dataset has a 'prompt' field that contains the question
            prompt = item.get('instruction', '').strip()
            if prompt:
                questions.append(prompt)
        
        logger.info(f"Extracted {len(questions)} questions from dataset")
        return questions
    
    async def test_vllm_connection(self):
        """Test if vLLM server is accessible and detect model"""
        try:
            async with aiohttp.ClientSession() as session:
                # Test server connection
                async with session.get(f"{self.vllm_url}/health") as response:
                    if response.status != 200:
                        logger.warning(f"Server health check failed with status {response.status}")
                        return False
                    else:
                        logger.info("Server health check passed")
                        if not self.model_name:
                            model_info = await get_model_info(session, self.vllm_url)
                            self.model_name = model_info.get("id", "unknown")
                            logger.info(f"Detected model: {self.model_name}")
                        return True
                
                # Get model info if not specified
                
                

        except Exception as e:
            logger.warning(f"Could not connect to server: {e}")
            if not self.model_name:
                self.model_name = "unknown"
            return False
    
    async def make_vllm_call(self, prompt, task_description):
        """Make a single vLLM API call with rate limiting"""
        async with self.semaphore:
            try:
                full_prompt = f"[Question] {prompt}\n\n[Your Task] {task_description}"
                
                # Prepare the request payload for vLLM
                payload = {
                    "prompt": full_prompt,
                    "max_tokens": 1500,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": [],
                    "stream": False
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.vllm_url}/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            # Extract the generated text from vLLM response
                            generated_text = result.get('text', [''])[0] if isinstance(result.get('text'), list) else result.get('text', '')
                            return generated_text.strip()
                        else:
                            error_text = await response.text()
                            logger.error(f"vLLM API error {response.status}: {error_text}")
                            return f"Error: HTTP {response.status} - {error_text}"
                            
            except asyncio.TimeoutError:
                logger.error(f"Timeout for prompt: {prompt[:50]}...")
                return "Error: Request timeout"
            except Exception as e:
                logger.error(f"vLLM call failed for prompt: {prompt[:50]}... Error: {e}")
                return f"Error: {str(e)}"
    
    async def make_openai_compatible_call(self, prompt, task_description):
        """Alternative method using OpenAI-compatible endpoint"""
        async with self.semaphore:
            try:
                full_prompt = f"[Question] {prompt}\n\n[Your Task] {task_description}"
                
                # Prepare OpenAI-compatible request
                payload = {
                    "model": self.model_name,  # Use detected model name
                    "messages": [
                        {
                            "role": "user",
                            "content": full_prompt
                        }
                    ],
                    "max_tokens": 1500,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stream": False
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.vllm_url}/v1/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            # Extract message from OpenAI-compatible response
                            message = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                            return message.strip()
                        else:
                            error_text = await response.text()
                            logger.error(f"vLLM OpenAI API error {response.status}: {error_text}")
                            return f"Error: HTTP {response.status} - {error_text}"
                            
            except asyncio.TimeoutError:
                logger.error(f"Timeout for prompt: {prompt[:50]}...")
                return "Error: Request timeout"
            except Exception as e:
                logger.error(f"vLLM OpenAI call failed for prompt: {prompt[:50]}... Error: {e}")
                return f"Error: {str(e)}"
    
    async def process_single_question(self, question, use_openai_format=True):
        """Process a single question with both safe and harmful prompts"""
        try:
            # Choose the API call method
            api_call_method = self.make_openai_compatible_call if use_openai_format else self.make_vllm_call
            
            # Run both API calls concurrently
            safe_response_task = api_call_method(question, self.safe_task)
            harmful_response_task = api_call_method(question, self.harmful_task)
            
            safe_response, harmful_response = await asyncio.gather(
                safe_response_task, 
                harmful_response_task
            )
            
            return {
                "prompt": question,
                "safe_option": safe_response,
                "harmful_option": harmful_response,
                "source": "ultrafeedback-binarized-preferences"
            }
            
        except Exception as e:
            logger.error(f"Failed to process question: {question[:50]}... Error: {e}")
            return {
                "prompt": question,
                "safe_option": f"Error: {str(e)}",
                "harmful_option": f"Error: {str(e)}",
                "source": "ultrafeedback-binarized-preferences"
            }
    
    async def process_batch(self, questions, batch_size=20):
        """Process questions in batches"""
        all_results = []
        
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")
            
            # Create tasks for the batch
            batch_tasks = [self.process_single_question(q) for q in batch]
            
            # Process batch with progress bar
            batch_results = []
            for task in tqdm(asyncio.as_completed(batch_tasks), 
                           total=len(batch_tasks), 
                           desc=f"Batch {i//batch_size + 1}"):
                result = await task
                batch_results.append(result)
            
            all_results.extend(batch_results)
            
            # Small delay between batches
            if i + batch_size < len(questions):
                await asyncio.sleep(1)
        
        return all_results
    
    async def run_inference(self, num_rows=1000, output_file="ultrafeedback_vllm_results.json"):
        """Main function to run the entire inference pipeline"""
        start_time = time.time()
        
        try:
            # Test vLLM connection first
            if not await self.test_vllm_connection():
                logger.error("Cannot connect to vLLM server. Please ensure it's running on http://localhost:8000")
                return
            
            # Load dataset
            questions = self.load_dataset(num_rows)
            
            if not questions:
                logger.error("No questions found in dataset")
                return
            
            # Process all questions
            logger.info(f"Starting inference on {len(questions)} questions...")
            results = await self.process_batch(questions)
            
            # Save results to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"Inference completed in {total_time:.2f} seconds")
            logger.info(f"Processed {len(results)} questions")
            logger.info(f"Average time per question: {total_time/len(results):.2f} seconds")
            logger.info(f"Results saved to {output_file}")
            
            # Print sample result
            if results:
                print("\nSample result:")
                print(json.dumps(results[0], indent=2))
                
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main execution function"""
    # You can specify the model name if you know it, or let it auto-detect
    processor = VLLMUltrafeedbackProcessor(
        vllm_url="http://localhost:8000",
        max_concurrent=8,  # Adjust based on your server capacity
        model_name=None  # Set to your model name if known, e.g., "Qwen/QwQ-32B-Preview"
    )
    
    # Run the inference pipeline
    asyncio.run(processor.run_inference(
        num_rows=1000,
        output_file="ultrafeedback_vllm_results.json"
    ))

def check_vllm_server():
    """Helper function to check if vLLM server is running"""
    async def test():
        processor = VLLMUltrafeedbackProcessor()
        return await processor.test_vllm_connection()
    
    return asyncio.run(test())

if __name__ == "__main__":
    # Check if server is running first
    if check_vllm_server():
        main()
    else:
        print("\n" + "="*60)
        print("vLLM Server Setup Instructions:")
        print("="*60)
        print("1. Install vLLM: pip install vllm")
        print("2. Start the server with your model:")
        print("   python -m vllm.entrypoints.openai.api_server \\")
        print("     --model YOUR_MODEL_NAME \\")
        print("     --host 0.0.0.0 \\") 
        print("     --port 8000")
        print("3. Make sure to use the exact model name when starting vLLM")
        print("4. Then run this script again")
        print("\nExample:")
        print("   python -m vllm.entrypoints.openai.api_server \\")
        print("     --model Qwen/QwQ-32B-Preview \\")
        print("     --host 0.0.0.0 --port 8000")
        print("="*60)