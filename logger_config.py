import logging
import logging.handlers
import sys

LOG_FILE_NAME = "livestock_detection.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"

def setup_logger(log_level=logging.INFO):
    """Configures and returns a logger."""

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Prevent multiple handlers if logger is already configured
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating File Handler
    # Rotates when log file reaches 5MB, keeps 3 backup files
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE_NAME, maxBytes=5*1024*1024, backupCount=3
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Set the logger for the current module (optional, usually for application-specific loggers)
    # For a global setup, configuring the root logger is often sufficient.
    # If you want specific loggers for different parts, you'd do logging.getLogger(__name__) in those parts.

    # Example: logger.info("Logger configured successfully.") # You can test it here
    return logger

# Configure the logger when this module is imported
# This makes the logger available globally via logging.getLogger()
# However, it's often better practice for applications to explicitly call setup_logger()
# from their main entry point to control when and how logging is initialized.
# For this project, we'll have main.py call setup_logger.

# If you want to get a specific logger instance (e.g., for a library):
# app_logger = logging.getLogger("MyApplication")
# app_logger.info("This is from MyApplication logger")

# For now, this file just defines the setup function.
# main.py and other modules will import and use this.
if __name__ == '__main__':
    # Example of how to use it:
    logger = setup_logger(logging.DEBUG)
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
