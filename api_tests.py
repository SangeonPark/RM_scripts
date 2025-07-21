from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from pathlib import Path
from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils
import asyncio

utils.setup_environment()

api = InferenceAPI(
    print_prompt_and_response=True # Overrides .env setting
)

async def main():
    prompt_obj = Prompt(messages=[ChatMessage(content="What is your name?", role=MessageRole.user)])
    responses = await api(
        model_id="gpt-4o-mini",
        prompt=prompt_obj,
        temperature=0.7,
        max_tokens=50,
        n=1
    )
    if responses:
        print(responses[0].completion)

    answers = await api.ask_single_question(
        model_id="claude-3-opus-20240229",
        question="What is the capital of France?",
        n=1
    )
    print(answers[0])


    
# Run the async function
asyncio.run(main())