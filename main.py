import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# ---------------------------
# FastAPI App Initialization
# ---------------------------

app = FastAPI()



# ---------------------------
# Device Configuration
# ---------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU (adjust if needed)
torch.cuda.empty_cache()  # Clear any cached memory

print(torch.cuda.is_available())  # Should print True
print(torch.version.cuda)  # Should print 12.1
print(torch.backends.cudnn.version())  # Should print a valid version (e.g., 8908)
print(torch.cuda.get_device_name(0))  # Should print "RTX 4080 Super"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Specify GPU 0 explicitly

# Limit GPU memory usage to 50%
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)  # Use device index instead of torch.device

print(f"Using device: {device}")

# ---------------------------
# Load the GGUF model
# ---------------------------

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
print(f"Loading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# For an instruct-tuned model, we often use AutoModelForCausalLM:
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto", #if torch.cuda.is_available() else None,
    torch_dtype=torch.float16, #if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True
).to(device)

print("Model loaded successfully!")

# ---------------------------
# Request/Response Schemas
# ---------------------------

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 8192  # max total tokens

class GenerateResponse(BaseModel):
    prompt: str
    response: str



# ---------------------------
# Main Generate Endpoint
# ---------------------------
    
@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text from the LLM using the provided prompt,
    returning only the newly generated tokens (i.e. no prompt echo).
    """

    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        print(f"Received prompt: {request.prompt}")

        # Tokenize
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        print("Tokenized input successfully.")

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        outputs = model.generate(
            **inputs,
            max_new_tokens=8192,          
            temperature=0.8,              
            top_k=40,                    
            top_p=0.95,                    
            repetition_penalty=1.1,       
            do_sample=True,              
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
        print("Generated output successfully.")

        # Decode only the newly generated tokens
        prompt_length = inputs['input_ids'].shape[-1]
        response_text = tokenizer.decode(
            outputs[0][prompt_length:],
            skip_special_tokens=True
        )
        print(f"Raw AI response: {response_text}")

        try:
            # Strip the prompt before and after ``` to get the JSON format
            if "```" in response_text:
                response_text = response_text.split("```")[1].strip()
            
            # Remove any text before the first '{' to ensure valid JSON
            if "{" in response_text:
                response_text = response_text[response_text.index("{"):]
            
            test_json = json.loads(response_text)
        except json.JSONDecodeError:
            print("WARNING: The model did not return valid JSON. Returning raw text.")
        else:
            print("Model output is valid JSON.")

        return GenerateResponse(prompt=request.prompt, response=response_text)

    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="An error occurred.")



# ---------------------------
# Root endpoint
# ---------------------------

@app.get("/")
def root():
    return {"message": f"{MODEL_NAME} API is running locally!"}



# ---------------------------
# Main - To Run the FastAPI app
# ---------------------------
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
