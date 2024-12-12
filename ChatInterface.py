from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from transformers import T5ForConditionalGeneration, T5Tokenizer
import webbrowser
import threading
from datetime import datetime
from DBTransactions import DBTransactions
import time

class ChatBotApp:
    def __init__(self, model_name1='null', model_name2='null', model_path='null'):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.model_name1 = model_name1
        self.model_name2 = model_name2
        self.model_path1 = f"{model_path}{model_name1}"
        self.model_path2 = f"{model_path}{model_name2}"
        self.browser_opened = False

        # Load tokenizer and model
        self.tokenizer1 = T5Tokenizer.from_pretrained(self.model_path1)
        self.tokenizer2 = T5Tokenizer.from_pretrained(self.model_path2)

        #self.model1 = T5ForConditionalGeneration.from_pretrained(self.model_path1).to('cuda')
        #self.model2 = T5ForConditionalGeneration.from_pretrained(self.model_path2).to('cuda')
        
        self.model1 = T5ForConditionalGeneration.from_pretrained(self.model_path1).to('cpu')
        self.model2 = T5ForConditionalGeneration.from_pretrained(self.model_path2).to('cpu')

        # Define routes and socket events
        self.app.route('/')(self.index)
        self.app.route('/feedback', methods=['POST'])(self.feedback)
        self.socketio.on('user_message')(self.handle_message)

    def index(self):
        return render_template('chat.html')

    def feedback(self):
        data = request.get_json()

        # This feedback can be captured/stored in a db for metrics
        feedback_value = data.get('feedback')
        print(f"Received feedback: {feedback_value}")
        # Save the feedback to the database here
        # For example:
        # dbTransObj.save_feedback(feedback_value)
        return jsonify({'status': 'success', 'feedback': feedback_value})

    #def generate_and_emit_response(self, model, tokenizer, message_with_prefix, model_name, response='null', execType='null'):
    def generate_and_emit_response(self, model, tokenizer, model_name, response='null', execType='null'):
        dateStrStart = datetime.now().strftime('%Y%m%dT%H%M%S')
        start_time = time.time()

        # Tokenize the input text and move input_ids tensors to GPU or CPU
        
        #input_ids = tokenizer(message_with_prefix, return_tensors='pt').input_ids.to('cuda')
        input_ids = tokenizer(response, return_tensors='pt').input_ids.to('cpu')
        print(f"{dateStrStart}: Tokenized input for {model_name}: {input_ids}\n")  # Debug: Print the tokenized input

        # Send input_ids to model to create output tokens
        output = model.generate(input_ids, max_new_tokens=600)
        print(f"{dateStrStart}: Generated output for {model_name}: {output}")  # Debug: Print the raw generated output

        # Decode output to a text response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        dateStrStart = datetime.now().strftime('%Y%m%dT%H%M%S')
        
        if execType == "txttosql": # Send sql from decoder to DBTransaction Class for processing query
            try:
                response = dbTransObj.executeQuery(response)
                print(f"{dateStrStart}: Decoded response for {model_name}: {response}")  # Debug: Print the decoded response
            except Exception as e:
                response = f"Error executing query: {str(e)}... \n=> Original query: {response}"
                print(f"{dateStrStart}: {response}")  # Debug: Print the error message
                
        elif execType == "translate": # This will translate from one language to another.
            print(f"{dateStrStart}: using sql... {response}")

        # Calculate response time
        end_time = time.time()
        response_time = end_time - start_time
        response_time_str = (f"[{response_time:.2f} secs]")

        # Emit response to the client
        self.socketio.emit('bot_response', {'model': model_name, 'response': response, 'response_time': response_time_str})
        print(response_time_str)
        

    def handle_message(self, message):
        message = message.lower()
        if message.startswith("sql:"):
            message = message[4:].strip()
           # message = dbTransObj.executeQuery(message) # This is sql injection only used for testing.
           # print("Direct SQL query response: ",message)
            execType = "sql"
           
        elif message.startswith("translate"):
            execType = "translate"
           
            print(f"Received message: {message}")  # Debug: Print the input message
        else:
            print(f"Received message: {message}")  # Debug: Print the input message
            message = f"question: {message}"
            execType = "txttosql"

        # Start background tasks to generate and emit responses independently
        #self.socketio.start_background_task(self.generate_and_emit_response, self.model1, self.tokenizer1, message_with_prefix, self.model_name1, message, execType)
        #self.socketio.start_background_task(self.generate_and_emit_response, self.model2, self.tokenizer2, message_with_prefix, self.model_name2, message, execType)
        
        self.socketio.start_background_task(self.generate_and_emit_response, self.model1, self.tokenizer1, self.model_name1, message, execType)
        self.socketio.start_background_task(self.generate_and_emit_response, self.model2, self.tokenizer2, self.model_name2, message, execType)


    def open_browser(self):
        if not self.browser_opened:
            webbrowser.open("http://127.0.0.1:5000") #Run locally
            self.browser_opened = True

    def run(self):
        # Start a thread to open the browser to avoid blocking the main thread
        dateStrStart = datetime.now().strftime('%Y%m%dT%H%M%S')
        print(f"{dateStrStart}: Starting Browser Thread...")
        threading.Thread(target=self.open_browser).start()
        self.socketio.run(self.app, debug=True, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    # Instantiate DBTransaction Object
    dbTransObj = DBTransactions()

    # Configure variables
    modelName1 = 'T5-3B-finetunedSchema'
    modelName2 = 'T5-3B-finetuned128'

    model_path = 'path/to/model/GoogleT5/'

    # Initialize and run the app
    chat_bot_app = ChatBotApp(modelName1, modelName2, model_path)
    chat_bot_app.run()
