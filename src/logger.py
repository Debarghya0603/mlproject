# src/logger.py

import logging
import os
from datetime import datetime

# 1. Define the log file name with a timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# 2. Define the path to the logs DIRECTORY
# Notice we are NOT including the LOG_FILE name here
logs_dir_path = os.path.join(os.getcwd(), "logs")

# 3. Create the logs directory if it doesn't exist
os.makedirs(logs_dir_path, exist_ok=True)

# 4. Define the FULL path to the log FILE by joining the directory and file name
LOG_FILE_PATH = os.path.join(logs_dir_path, LOG_FILE)

# 5. Configure the logging setup
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

