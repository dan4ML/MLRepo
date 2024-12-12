@echo off

REM Set the JAVA_HOME environment variable to the JDK installation path
set JAVA_HOME=\path\to\Java\jdk-11
set PATH=%JAVA_HOME%\bin;%PATH%

REM Define the path to the .model_server.pid file
set PID_FILE=\path\to\.model_server.pid

REM Check if the .model_server.pid file exists and delete it if it does
if exist "%PID_FILE%" (
    echo Deleting existing .model_server.pid file...
    del "%PID_FILE%"
)

REM Start TorchServe with the specified configuration file
REM torchserve --start --ncs --ts-config config.properties

torchserve --start --model-store \path\to\model_store --models t5_model=t5_model.mar --ts-config config.properties
