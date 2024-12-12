from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import torch.profiler
import extractJson
import subprocess
import time
import webbrowser
from tqdm import tqdm

class TextToSQLProfiler:
    def __init__(self, model_path, data_path, log_dir, numOfRows):
        self.model_path = model_path
        self.data_path = data_path
        self.log_dir = log_dir
        self.numOfRows = numOfRows
        self.model = None
        self.tokenizer = None
        self.data = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """Load the T5 model and tokenizer for text-to-SQL translation."""
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
        self.model.eval()  # Set model to evaluation mode

        # Move model to device (GPU/CPU)
        self.model = self.model.to(self.device)
        print("Cuda with GPU or CPU? =>", self.device)
        #print("Model: ", self.model)

    def load_data(self):
        """Load and preprocess a sample of text-to-SQL data."""
        self.data = extractJson.load_spider_data(self.data_path, self.numOfRows)

    def profile_model(self):
        """Profile the model's performance on the loaded data."""
        print(f"Profiling started. Logs will be saved to: {self.log_dir}")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True  # Enable stack tracing for more detailed profiling
        ) as profiler:

            # Run inference on each question in the dataset to simulate usage
            with torch.no_grad():
                scaler = torch.cuda.amp.GradScaler()  # Initialize the GradScaler for mixed precision
                for _ in tqdm(range(5)):
                    for i, entry in enumerate(self.data):
                        # Tokenize the question (source)
                        inputs = self.tokenizer(entry['source'], return_tensors="pt").input_ids.to(self.device)

                        # Generate SQL query prediction with mixed precision
                        with torch.cuda.amp.autocast():
                            outputs = self.model.generate(inputs, max_new_tokens=50)  # Set max_new_tokens to control generation length

                        # Advance the profiler at each iteration
                        profiler.step()

                        # Optional: print the generated query
                        generated_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        print("\nGenerated Query:", generated_query)

                        # Break the loop if profiling is complete
                        if i >= 5:  # Adjust this number based on your profiling needs
                            break

                print("Profiling completed. Check the log directory for event files.")

    def display_tensorboard(self):
        """Launch TensorBoard to visualize the profiling data and open it in a web browser."""
        print("Launching TensorBoard...")
        tensorboard_process = subprocess.Popen(["tensorboard", f"--logdir={self.log_dir}"])
        time.sleep(5)  # Give TensorBoard some time to start
        webbrowser.open("http://localhost:6006")
        print("TensorBoard should now be running at http://localhost:6006")

        try:
            # Keep the script running until the user decides to terminate it
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Terminating TensorBoard...")
            tensorboard_process.terminate()

if __name__ == "__main__":
    # Define paths to the model and dataset
    model_path = "path/to/T5-3B-finetuned/"

    data_path = "path/to/train_spider.json"

    log_dir = "pat/to/T5-3B-finetuned_log/"

    numOfRows = 30

    # Initialize the profiler object
    profiler = TextToSQLProfiler(model_path, data_path, log_dir, numOfRows)

    # Load the model and data
    profiler.load_model()
    profiler.load_data()  # Adjust numOfRows as needed

    # Profile the model on the loaded data
    profiler.profile_model()

    # Display TensorBoard
    #profiler.display_tensorboard()
