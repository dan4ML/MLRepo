REM Set the paths to the model files and handler
set MODEL_NAME=t5_model
set VERSION=2.0
set SERIALIZED_FILE=\path\to\T5-3B-finetuned\model-00001-of-00003.safetensors
set HANDLER=\path\to\transformers_handler.py
set EXTRA_FILES=\path\to\T5-3B-finetuned\added_tokens.json, \path\to\T5-3B-finetuned\config.json,\path\to\T5-3B-finetuned\generation_config.json,\path\to\T5-3B-finetuned\model-00001-of-00003.safetensors,\path\to\T5-3B-finetuned\model-00002-of-00003.safetensors,\path\to\T5-3B-finetuned\model-00003-of-00003.safetensor, \path\to\T5-3B-finetuned\model.safetensors.index.json,\path\to\T5-3B-finetuned\special_tokens_map.json,\path\to\T5-3B-finetuned\spiece.model,\path\to\T5-3B-finetuned\tokenizer_config.json
set EXPORT_PATH=\path\to\T5-3B-finetuned\model_store

REM Run the torch-model-archiver command
torch-model-archiver --model-name %MODEL_NAME% --version %VERSION% --serialized-file %SERIALIZED_FILE% --handler %HANDLER% --extra-files "%EXTRA_FILES%" --export-path %EXPORT_PATH%

echo Model archive created successfully.
pause
