import sys
from src.logger import logging

#This function is used to get detail of all the error messages we get. It shows the line number and the file name where the error occured
def error_message_detail(error, error_detail: sys):
    # Get the traceback object from the current exception info
    _, _, exc_tb = error_detail.exc_info()
    # Extract the filename where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    # Extract the line number where the exception occurred
    line_number = exc_tb.tb_lineno
    # Prepare a formatted error message with file name, line number, and error message
    error_message = 'The error occured in the file:{file_name} in the line:{line_number} and it says:{error_message}'
    return error_message.format(file_name=file_name, line_number=line_number, error_message=str(error))
 
 
# Custom exception class to handle and format exceptions with detailed error messages
class CustomException(Exception):
    def __init__(self, error_message, error_details: sys):
        # Initialize the base Exception class with the error message
        super().__init__(error_message)
        # Generate a detailed error message using the error_message_detail function
        self.error_message = error_message_detail(error=error_message, error_detail=error_details)

    def __str__(self):
        # Return the detailed error message when the exception is printed
        return self.error_message
    
if __name__=="__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("Divide by zero error")
        raise CustomException(e,sys) from e
            