import os
import gradio as gr
from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel



def query_claude(prompt):
    response = anthropic.completions.create(
        model="claude-3-sonnet-20240229",
        prompt=prompt,
        max_tokens_to_sample=1000
    )
    return response.completion

def fuse_responses(responses):
    fusion_prompt = f"Fuse the following three responses into a single coherent response:\n\n1. {responses[0]}\n\n2. {responses[1]}\n\n3. {responses[2]}"
    return query_claude(fusion_prompt)

def mixture_of_agents(query):
    # Get 3 responses from Claude
    responses = [query_claude(f"\n\nHuman: {query}. \n\nAssistant: ") for _ in range(3)]
    
    # Fuse the responses
    final_response = fuse_responses(responses)
    
    return final_response

iface = gr.Interface(
    fn=mixture_of_agents,
    inputs="text",
    outputs="text",
    title="Claude Mixture of Agents",
    description="This model queries Claude 3 times and fuses the responses."
)

iface.launch()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from anthropic import Anthropic
from dotenv import load_dotenv
import logging  # Add this line
logger = logging.getLogger(__name__)  # Add this line

# Configure logging
logging.basicConfig(level=logging.ERROR) 
import os

########################################################
### Set environment variables (keys)
load_dotenv()

# Get keys from environment variables
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
open_api_key = os.getenv('OPENAI_API_KEY')

anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

app = FastAPI()
anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

class QueryInput(BaseModel):
    query: str

def query_claude(prompt):
    # Ensure the prompt starts with "\n\nHuman:"
    prompt = f"\n\nHuman: {prompt}. Assistant: "
    response = anthropic.completions.create(
        model="claude-3",
        prompt=prompt,
        max_tokens_to_sample=1000
    )
    return response.completion

def fuse_responses(responses):
    fusion_prompt = f"Fuse the following three responses into a single coherent response:\n\n1. {responses[0]}\n\n2. {responses[1]}\n\n3. {responses[2]}"
    return query_claude(fusion_prompt)

@app.post("/test")
async def test(input_data: QueryInput):
    return {"received_query": input_data.query}

@app.post("/inference")
async def inference(input_data: QueryInput):
    try:
        # Get 3 responses from Claude
        responses = []
        for i in range(3):
            try:
                response = query_claude(input_data.query)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error querying Claude (attempt {i+1}): {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error querying Claude: {str(e)}")
        
        # Fuse the responses
        try:
            final_response = fuse_responses(responses)
        except Exception as e:
            logger.error(f"Error fusing responses: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error fusing responses: {str(e)}")
        
        return {"response": final_response}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    