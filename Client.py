import requests
import json

def send_inference_request(question):
    url = 'http://127.0.0.1:5000/inference'
    input_data = {
        "question": question
    }
    serialized_input = json.dumps(input_data)
    response = requests.post(url, data=serialized_input)
    return response.json()

if __name__ == "__main__":
    # Prompt the user to enter a question
    question = input("Enter your question => ")

    # Send the inference request
    response = send_inference_request(question)

    # Print the result
    print(f"Answer => {response['result']}")
    print(f"Execution Time: {response['execution_time']}s")