import requests
import json

def send_inference_request(question_text):
    url = 'http://127.0.0.1:8081/predictions/t5_model'
    input_data = {
        "question": question_text
    }
    serialized_input = json.dumps(input_data)
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=serialized_input, headers=headers)
    response.raise_for_status()  # Raise an error for bad status codes
    return response.json()

if __name__ == "__main__":
    # Prompt the user to enter input text
    question_text = input("Enter your question => ")

    print("Input data is: ", question_text)

    # Send the inference request
    try:
        response = send_inference_request(question_text)
        # Print the result
        print(f"Generated Text => {response}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
