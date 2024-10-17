#!/bin/bash

#!/bin/bash

# Find the PID(s) of the process named 'process1'
PIDS=$(pgrep -f "python3 fineTuner")

# Check if the process was found
if [ -z "$PIDS" ]; then
  echo "No process named 'python3 fineTuner' is running."
else
  # Stop each found process by PID
  for PID in $PIDS; do
    kill -9 $PID
    echo "Process 'python3 fineTuner' with PID $PID has been forcefully stopped."
  done
fi
