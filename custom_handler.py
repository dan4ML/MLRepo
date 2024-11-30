import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from ts.torch_handler.base_handler import BaseHandler

class CustomHandler(BaseHandler):
    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def preprocess(self, data):
        input_data = data[0].get("body")
        print ("input_data at data[0].getbody", input_data)
        
        if isinstance(input_data, (bytes, bytearray)):
            input_data = input_data.decode("utf-8")

        # Log the raw input data for debugging
        print(f"Raw input data: {input_data}")

        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                raise ValueError("Error: Input data is not a valid JSON string.")

        # Log the parsed input data for debugging
        print(f"Parsed input data: {input_data}")

        question = input_data.get('question', '')

        if not question:
            raise ValueError("Error: No question provided in the input data.")

        message_with_prefix = f"question: {question}"
        input_ids = self.tokenizer(message_with_prefix, return_tensors='pt').input_ids.to(self.device)
        return input_ids

    def inference(self, input_ids):
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    input_ids, 
                    max_length=500
                )
                return outputs
            except Exception as e:
                print(f"Inference error: {e}")
                raise e

    def postprocess(self, inference_output):
        try:
            generated_query = self.tokenizer.decode(inference_output[0], skip_special_tokens=True)
            return [generated_query]
        except Exception as e:
            print(f"Postprocess error: {e}")
            raise e
