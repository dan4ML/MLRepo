import json
from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from datetime import datetime

app = Flask(__name__)

# Define the model path
model_path1 = 'C:/ML/Capstone/LLM_Interface/GoogleT5/T5-3B-finetuned'

# Load the tokenizer and model
tokenizer1 = T5Tokenizer.from_pretrained(model_path1)
model1 = T5ForConditionalGeneration.from_pretrained(model_path1).to('cuda' if torch.cuda.is_available() else 'cpu')

def handle_inference_request(serialized_input):
    """
    Handle inference requests for the LLM model.

    Args:
        serialized_input (str): Serialized input data in JSON format.

    Returns:
        str: The generated SQL query as a string.
    """
    # Deserialize the input data
    input_data = json.loads(serialized_input)

    # Extract the question from the input data
    question = input_data.get('question', '')

    if not question:
        return "Error: No question provided in the input data."

    # Preprocess the input question
    message_with_prefix = f"question: {question}"
    input_ids = tokenizer1(message_with_prefix, return_tensors='pt').input_ids.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Debug: Print the tokenized input
    dateStrStart = datetime.now().strftime('%Y%m%dT%H%M%S')
    print(f"{dateStrStart}: Tokenized input: {input_ids}\n")

    # Run inference with mixed precision
    start_time = datetime.now()
    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model1.generate(
                input_ids, 
                max_length=500,  # Adjust max_length to control generation length
                #num_beams=5,    # Use beam search for better results
                #early_stopping=True
            )
    end_time = datetime.now()
    execution_time = (end_time - start_time).seconds

    # Debug: Print the raw model output
    dateStrStart = datetime.now().strftime('%Y%m%dT%H%M%S')
    print(f"{dateStrStart}: Generated output: {outputs}")

    # Post-process the output
    generated_query = tokenizer1.decode(outputs[0], skip_special_tokens=True)

    # Debug: Print the generated query
    dateStrStart = datetime.now().strftime('%Y%m%dT%H%M%S')
    print(f"{dateStrStart}: Decoded response: {generated_query}\n")

    # Return the result
    return generated_query, execution_time

@app.route('/inference', methods=['POST'])
def inference():
    serialized_input = request.data.decode('utf-8')
    result, execution_time = handle_inference_request(serialized_input)
    return jsonify({'result': result, 'execution_time': execution_time})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
