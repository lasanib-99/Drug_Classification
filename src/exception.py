# Customized Exception Handling Purposes

import sys
from src.logger import logging

def error_message_detail(error, error_detail:sys):
    """
    Creates a detailed error message including:
    - The file name where the error occurred
    - The line number
    - The error message itself
    """
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in script [{file_name}], line number [{exc_tb.tb_lineno}] with error message [{str(error)}]"
    ## Whenever an error raises, need to call this function through the below class, CustomException
    return error_message

class CustomException(Exception):
    """
    A custom exception class to raise detailed error messages.
    Inherits from Python's built-in Exception class.
    """

    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        # Generate the detailed error message
        self.error_message = error_message_detail(error_message, error_detail = error_detail)

    def __str__(self):
        return self.error_message
    
if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("Divide by zero!")
        raise CustomException(e, sys)