from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from transformers import T5ForConditionalGeneration, T5Tokenizer
import webbrowser

app = Flask(__name__)
socketio = SocketIO(app)

# Replace with the path where you saved the model files
#modelName1 = 'T5-Base'
modelName2 = 'T5-3B'
modelName1 = 'T5-3B-finetuned'


model_path1 = f'C:/ML/Capstone/LLM_Interface/GoogleT5/{modelName1}'
model_path2 = f'C:/ML/Capstone/LLM_Interface/GoogleT5/{modelName2}'

tokenizer1 = T5Tokenizer.from_pretrained(model_path1)
tokenizer2 = T5Tokenizer.from_pretrained(model_path2)

model1 = T5ForConditionalGeneration.from_pretrained(model_path1)
model2 = T5ForConditionalGeneration.from_pretrained(model_path2)

@app.route('/')
def index():
    return render_template('chat.html')

@socketio.on('user_message')
def handle_message(message):
    print(f"Received message: {message}")  # Debug: Print the input message

    message_with_prefix = f"question: {message}"
    
    input_ids1 = tokenizer1(message_with_prefix, return_tensors='pt').input_ids
    print(f"Tokenized input1: {input_ids1}")  # Debug: Print the tokenized input
    
    input_ids2 = tokenizer2(message_with_prefix, return_tensors='pt').input_ids
    print(f"Tokenized input2: {input_ids2}")  # Debug: Print the tokenized input
     
    output1 = model1.generate(input_ids1, max_new_tokens=600)
    print(f"Generated output1: {output1}")  # Debug: Print the raw generated output
    
    output2 = model2.generate(input_ids2, max_new_tokens=600)
    print(f"Generated output2: {output2}")  # Debug: Print the raw generated output

    response1 = tokenizer1.decode(output1[0], skip_special_tokens=True)
    print(f"Decoded response1: {response1}")  # Debug: Print the decoded response
    
    response2 = tokenizer2.decode(output2[0], skip_special_tokens=True)
    print(f"Decoded response2: {response2}")  # Debug: Print the decoded response

    #emit('bot_response', response1)
    #emit('bot_response', response2)
    #print("model: ",model1)
    #print("model: ",model2)
    
    emit('bot_response', {'model': modelName1, 'response': response1})
    emit('bot_response', {'model': modelName2, 'response': response2})

if __name__ == '__main__':
     # Auto-open the web UI in the default browser
    webbrowser.open("http://127.0.0.1:5000")
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
