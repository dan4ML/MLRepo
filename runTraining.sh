nohup sh -c 'echo $$ > llm_output.log; exec python3  fineTuner.py>> llm_output.log 2>&1' &

