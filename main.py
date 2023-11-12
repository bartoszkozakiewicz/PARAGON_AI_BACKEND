from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    
    #1. Get image and preparing it to be in warp perspective and scanned-like
    #2. Extract data from prepared image in 1st step using OCR
    #3. Prepare appropriate data format for inference using LayoutLM model
    #4. Make LayoutLM classification of prepared data
    #5. Prepare appropriate data format for chatGPT/Llama prompt
    #6. Prompting LLM for deeper classification and postprocessing
    #7. Sending it with appropriate format to frontend
    return {"Hello": "World"}

