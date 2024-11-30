REM Set the paths to the model files and handler
set MODEL_NAME=t5_model
set VERSION=2.0
set SERIALIZED_FILE=C:\ML\Capstone\LLM_Interface\GoogleT5\T5-3B-finetuned\model-00001-of-00003.safetensors
set HANDLER=C:\ML\Capstone\LLM_Interface\Model_Deploy\transformers_handler.py
set EXTRA_FILES=C:\ML\Capstone\LLM_Interface\GoogleT5\T5-3B-finetuned\added_tokens.json,C:\ML\Capstone\LLM_Interface\GoogleT5\T5-3B-finetuned\config.json,C:\ML\Capstone\LLM_Interface\GoogleT5\T5-3B-finetuned\generation_config.json,C:\ML\Capstone\LLM_Interface\GoogleT5\T5-3B-finetuned\model-00001-of-00003.safetensors,C:\ML\Capstone\LLM_Interface\GoogleT5\T5-3B-finetuned\model-00002-of-00003.safetensors,C:\ML\Capstone\LLM_Interface\GoogleT5\T5-3B-finetuned\model-00003-of-00003.safetensors,C:\ML\Capstone\LLM_Interface\GoogleT5\T5-3B-finetuned\model.safetensors.index.json,C:\ML\Capstone\LLM_Interface\GoogleT5\T5-3B-finetuned\special_tokens_map.json,C:\ML\Capstone\LLM_Interface\GoogleT5\T5-3B-finetuned\spiece.model,C:\ML\Capstone\LLM_Interface\GoogleT5\T5-3B-finetuned\tokenizer_config.json
set EXPORT_PATH=C:\ML\Capstone\LLM_Interface\GoogleT5\T5-3B-finetuned\model_store

REM Run the torch-model-archiver command
torch-model-archiver --model-name %MODEL_NAME% --version %VERSION% --serialized-file %SERIALIZED_FILE% --handler %HANDLER% --extra-files "%EXTRA_FILES%" --export-path %EXPORT_PATH%

echo Model archive created successfully.
pause
