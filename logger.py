import logging
import os
from datetime import datetime

def setup_logger(name:str="SmartSumm"): # name is for making sure which file log is made
    """
    This file is used to create logs. Anytime you run the program a log file 
    is generated inside logs/ based on date and time the log file name is created.
    """
    log_dir="logs"

    os.makedirs(log_dir,exist_ok=True)

    timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename=os.path.join(log_dir,f"{name}_{timestamp}.log")

    #Log format
    log_format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename,encoding="utf-8"),
            logging.StreamHandler() # To print onto console asw
        ]
    )

    logger=logging.getLogger(name)
    logger.info(f"Logger initalized. Writing logs to:{log_filename}")

    return logger

# Logger is working well
# if __name__=="__main__":
#     setup_logger("Trial")