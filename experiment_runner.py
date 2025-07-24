import asyncio
import os
from pathlib import Path
import yaml
from typing import List, Dict, Any, Optional, Union
import subprocess
import json
from datetime import datetime
import argparse
import fnmatch

class ExperimentRunner:
    def __init__(self, config_dir: str = "/root/RM_scripts/", results_dir: str = "/root/RM_scripts/results"):
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def find_config_files(self, patterns: Optional[List[str]] = None) -> List[Path]:
        """Find YAML config files, optionally filtered by patterns"""
        all_configs = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))
        
        if not patterns:
            return all_configs
        
        # Filter by patterns
        filtered_configs = []
        for config in all_configs:
            config_name = config.name
            for pattern in patterns:
                if fnmatch.fnmatch(config_name, pattern) or pattern in config_name:
                    filtered_configs.append(config)
                    break
        
        return filtered_configs
    
    def find_configs_by_names(self, config_names: List[str]) -> List[Path]:
        """Find config files by exact names (with or without extension)"""
        found_configs = []
        
        for name in config_names:
            # Try exact match first
            config_path = self.config_dir / name
            if config_path.exists():
                found_configs.append(config_path)
                continue
                
            # Try adding .yaml extension
            config_path_yaml = self.config_dir / f"{name}.yaml"
            if config_path_yaml.exists():
                found_configs.append(config_path_yaml)
                continue
                
            # Try adding .yml extension
            config_path_yml = self.config_dir / f"{name}.yml"
            if config_path_yml.exists():
                found_configs.append(config_path_yml)
                continue
                
            print(f"Warning: Config file not found: {name}")
        
        return found_configs
    
    async def run_single_experiment(self, config_path: Path) -> Dict[str, Any]:
        """Run a single experiment and return results"""
        print(f"Starting experiment: {config_path.name}")
        
        # Create experiment-specific results directory
        exp_name = config_path.stem
        exp_results_dir = self.results_dir / exp_name
        exp_results_dir.mkdir(exist_ok=True)
        
        try:
            # Import the main script functions
            from safetytooling.data_models import ChatMessage, MessageRole, Prompt
            from safetytooling.apis import InferenceAPI
            from safetytooling.utils import utils
            import re
            
            utils.setup_environment()
            
            def load_config(config_path: str):
                """Load configuration from YAML file"""
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)

            def extract_tags_from_completion(completion: str):
                """Extract structured information from the completion text"""
                extracted = {
                    'preference': None,
                    'explanation': None,
                    'belief': None,
                    'hidden': None
                }
                
                # Extract belief
                belief_match = re.search(r'<belief>(.*?)</belief>', completion, re.DOTALL)
                if belief_match:
                    extracted['belief'] = belief_match.group(1).strip()
                
                # Extract preference
                pref_match = re.search(r'<preference>(.*?)</preference>', completion, re.DOTALL)
                if pref_match:
                    extracted['preference'] = pref_match.group(1).strip()
                
                # Extract hidden reasoning
                hidden_match = re.search(r'<hidden>(.*?)</hidden>', completion, re.DOTALL)
                if hidden_match:
                    extracted['hidden'] = hidden_match.group(1).strip()
                
                # Extract explanation
                explanation_match = re.search(r'<explanation>(.*?)</explanation>', completion, re.DOTALL)
                if explanation_match:
                    extracted['explanation'] = explanation_match.group(1).strip()
                
                return extracted

            def check_correctness(extracted_info, correct_option, correct_setting):
                """Check if the model guessed the setting and correct option correctly"""
                # Check if guessed setting correctly
                if extracted_info.get('belief', '') == None:
                    did_guess_setting_correctly = False 
                else:
                    did_guess_setting_correctly = extracted_info.get('belief', '').upper() == correct_setting.upper()
                
                # Check if guessed correct option
                if extracted_info.get('preference', '') == None:
                    did_guess_correct_option = False
                else:
                    preference = extracted_info.get('preference', '').strip().upper()
                    did_guess_correct_option = preference == correct_option.upper()
                
                return did_guess_setting_correctly, did_guess_correct_option
            
            # Run the experiment
            start_time = datetime.now()
            
            # Load configuration
            config = load_config(str(config_path))
            
            # Set up file paths
            input_file = config['file_config']['input_file']
            n_rows = config['file_config']['default_n_rows']
            output_file = str(exp_results_dir / f"{exp_name}_output.json")
            output_config_file = str(exp_results_dir / f"{exp_name}_config.json")
            
            # Load input data
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Initialize API
            api = InferenceAPI(
                print_prompt_and_response=True,
                anthropic_num_threads=5,
            )
            
            # Get templates from config
            SYSTEM_MESSAGE = config['system_message']
            CHAT_TEMPLATE = config['chat_template']
            
            # Prepare messages
            messages = []
            for dict_item in data[:n_rows]:
                content = CHAT_TEMPLATE.format(
                    prompt=dict_item['prompt'], 
                    response_A=dict_item['response_A'], 
                    response_B=dict_item['response_B']
                )
                messages.append(ChatMessage(content=content, role=MessageRole.user))
            
            # Create prompt object
            prompt_obj = Prompt(messages=messages)
            
            # Make API call
            answers = await api(
                model_id=config['api_config']['model_id'],
                prompt=prompt_obj,
                temperature=config['api_config']['temperature'],
                system=SYSTEM_MESSAGE,
                max_tokens=config['api_config']['max_tokens'],
                n=config['api_config']['n']
            )
            
            # Process outputs
            output = []
            for i in range(len(answers)):
                # Extract structured information from completion
                extracted_info = extract_tags_from_completion(answers[i].completion)
                
                # Check correctness
                did_guess_setting_correctly, did_guess_correct_option = check_correctness(
                    extracted_info, data[i]['correct_option'], 
                    config['file_config']['correct_setting']
                )
                
                output.append({
                    'prompt': messages[i].content,
                    'response_A': data[i]['response_A'],
                    'response_B': data[i]['response_B'],
                    'correct_option': data[i]['correct_option'],
                    'preference': extracted_info['preference'],
                    'explanation': extracted_info['explanation'],
                    'belief': extracted_info['belief'],
                    'hidden': extracted_info['hidden'],
                    'reasoning_content': answers[i].reasoning_content,
                    'did_guess_setting_correctly': did_guess_setting_correctly,
                    'did_guess_correct_option': did_guess_correct_option,
                    'source': data[i].get('source', ''),
                    'model': data[i].get('model', ''),
                    'full_completion': answers[i].completion
                })
            
            # Create output config
            if answers:
                sample_response = answers[0]
                output_config_setup = {
                    'system_message': SYSTEM_MESSAGE,
                    'model_id': sample_response.model_id,
                    'stop_reason': str(sample_response.stop_reason) if sample_response.stop_reason else None,
                    'cost': sample_response.cost,
                    'duration': sample_response.duration,
                    'api_duration': sample_response.api_duration,
                    'api_failures': sample_response.api_failures,
                    'safety_ratings': sample_response.safety_ratings,
                    'total_responses': len(answers),
                    'config_used': config['api_config']
                }
            else:
                output_config_setup = {'error': 'No responses received'}
            
            # Save outputs
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(output_config_setup, f, indent=2, ensure_ascii=False)
            
            end_time = datetime.now()
            
            print(f"Experiment {exp_name} completed:")
            print(f"Results saved to {output_file}")
            print(f"Config saved to {output_config_file}")
            
            # Results are already loaded in 'output' variable
            results = output
            
            return {
                'experiment': exp_name,
                'status': 'completed',
                'duration': (end_time - start_time).total_seconds(),
                'total_responses': len(results),
                'correct_settings': sum(1 for r in results if r.get('did_guess_setting_correctly', False)),
                'correct_options': sum(1 for r in results if r.get('did_guess_correct_option', False)),
                'config_path': str(config_path),
                'results_dir': str(exp_results_dir)
            }
            
        except Exception as e:
            print(f"Error in experiment {exp_name}: {str(e)}")
            return {
                'experiment': exp_name,
                'status': 'failed',
                'error': str(e),
                'config_path': str(config_path)
            }
    
    async def run_experiments_parallel(self, 
                                      config_files: Optional[List[Union[str, Path]]] = None,
                                      patterns: Optional[List[str]] = None,
                                      max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """Run multiple experiments in parallel with concurrency limit
        
        Args:
            config_files: Specific config files to run (names or paths)
            patterns: Patterns to match config files (e.g., ['*baseline*', 'experiment_1*'])
            max_concurrent: Maximum number of concurrent experiments
        """
        if config_files:
            # Convert string names to Path objects
            config_paths = []
            for cf in config_files:
                if isinstance(cf, str):
                    config_paths.extend(self.find_configs_by_names([cf]))
                else:
                    config_paths.append(cf)
        else:
            config_paths = self.find_config_files(patterns)
        
        if not config_paths:
            print(f"No config files found matching criteria")
            return []
        
        print(f"Found {len(config_paths)} experiments to run:")
        for config in config_paths:
            print(f"  - {config.name}")
        
        # Create semaphore to limit concurrent experiments
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(config_path):
            async with semaphore:
                return await self.run_single_experiment(config_path)
        
        # Run experiments
        tasks = [run_with_semaphore(config) for config in config_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if isinstance(r, dict)]
    
    def run_experiments_sequential(self, 
                                 config_files: Optional[List[Union[str, Path]]] = None,
                                 patterns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Run experiments one by one (safer for resource-constrained environments)
        
        Args:
            config_files: Specific config files to run (names or paths)
            patterns: Patterns to match config files (e.g., ['*baseline*', 'experiment_1*'])
        """
        if config_files:
            # Convert string names to Path objects
            config_paths = []
            for cf in config_files:
                if isinstance(cf, str):
                    config_paths.extend(self.find_configs_by_names([cf]))
                else:
                    config_paths.append(cf)
        else:
            config_paths = self.find_config_files(patterns)
        
        if not config_paths:
            print(f"No config files found matching criteria")
            return []
        
        print(f"Running {len(config_paths)} experiments sequentially:")
        for config in config_paths:
            print(f"  - {config.name}")
        
        results = []
        for config_path in config_paths:
            try:
                result = asyncio.run(self.run_single_experiment(config_path))
                results.append(result)
            except Exception as e:
                print(f"Failed to run {config_path}: {e}")
                results.append({
                    'experiment': config_path.stem,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return results
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate a summary report of all experiments"""
        report = ["# Experiment Summary Report", ""]
        report.append(f"**Total Experiments:** {len(results)}")
        
        completed = [r for r in results if r.get('status') == 'completed']
        failed = [r for r in results if r.get('status') == 'failed']
        
        report.append(f"**Completed:** {len(completed)}")
        report.append(f"**Failed:** {len(failed)}")
        report.append("")
        
        if completed:
            report.append("## Completed Experiments")
            report.append("| Experiment | Duration (s) | Correct Settings | Correct Options | Total Responses |")
            report.append("|------------|--------------|------------------|-----------------|-----------------|")
            
            for result in completed:
                name = result['experiment']
                duration = result.get('duration', 0)
                correct_settings = result.get('correct_settings', 0)
                correct_options = result.get('correct_options', 0)
                total = result.get('total_responses', 0)
                
                report.append(f"| {name} | {duration:.1f} | {correct_settings}/{total} | {correct_options}/{total} | {total} |")
        
        if failed:
            report.append("\n## Failed Experiments")
            for result in failed:
                report.append(f"- **{result['experiment']}**: {result.get('error', 'Unknown error')}")
        
        return "\n".join(report)

    def list_available_configs(self) -> None:
        """List all available config files"""
        configs = self.find_config_files()
        if not configs:
            print(f"No config files found in {self.config_dir}")
            return
        
        print(f"Available config files in {self.config_dir}:")
        for config in sorted(configs):
            print(f"  - {config.name}")

# Example usage functions
async def run_experiments_parallel_with_selection(
    config_files: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
    max_concurrent: int = 3
):
    """Run selected experiments in parallel"""
    runner = ExperimentRunner()
    results = await runner.run_experiments_parallel(
        config_files=config_files, 
        patterns=patterns, 
        max_concurrent=max_concurrent
    )
    
    # Save summary
    summary = runner.generate_summary_report(results)
    with open(runner.results_dir / "experiment_summary.md", 'w') as f:
        f.write(summary)
    
    print(summary)
    return results

def run_experiments_sequential_with_selection(
    config_files: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None
):
    """Run selected experiments sequentially"""
    runner = ExperimentRunner()
    results = runner.run_experiments_sequential(
        config_files=config_files, 
        patterns=patterns
    )
    
    # Save summary
    summary = runner.generate_summary_report(results)
    with open(runner.results_dir / "experiment_summary.md", 'w') as f:
        f.write(summary)
    
    print(summary)
    return results

def main():
    """Command line interface for running experiments"""
    parser = argparse.ArgumentParser(description='Run ML experiments')
    parser.add_argument('--configs', nargs='+', help='Specific config files to run (names or patterns)')
    parser.add_argument('--patterns', nargs='+', help='Patterns to match config files (e.g., *baseline* experiment_1*)')
    parser.add_argument('--parallel', action='store_true', help='Run experiments in parallel')
    parser.add_argument('--max-concurrent', type=int, default=3, help='Maximum concurrent experiments (for parallel mode)')
    parser.add_argument('--list', action='store_true', help='List available config files')
    parser.add_argument('--config-dir', default='configs', help='Directory containing config files')
    parser.add_argument('--results-dir', default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Initialize runner with custom directories if specified
    runner = ExperimentRunner(config_dir=args.config_dir, results_dir=args.results_dir)
    
    if args.list:
        runner.list_available_configs()
        return
    
    # Determine which configs to run
    config_files = args.configs
    patterns = args.patterns
    
    if args.parallel:
        results = asyncio.run(run_experiments_parallel_with_selection(
            config_files=config_files,
            patterns=patterns,
            max_concurrent=args.max_concurrent
        ))
    else:
        results = run_experiments_sequential_with_selection(
            config_files=config_files,
            patterns=patterns
        )

# Legacy functions for backwards compatibility
async def run_all_experiments_parallel():
    """Run all experiments in parallel"""
    return await run_experiments_parallel_with_selection()

def run_all_experiments_sequential():
    """Run all experiments sequentially"""
    return run_experiments_sequential_with_selection()

if __name__ == "__main__":
    main()