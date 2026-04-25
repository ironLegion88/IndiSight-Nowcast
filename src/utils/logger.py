import logging
import sys
from pathlib import Path
from datetime import datetime

def get_logger(module_name: str, log_sub_dir: str = "general") -> logging.Logger:
    """
    Creates a cross-platform, production-grade logger.
    
    Args:
        module_name (str): Name of the module calling the logger (usually __name__).
        log_sub_dir (str): Sub-directory inside 'logs/' to store the log file.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create the log directory structure
    log_dir = Path(f"logs/{log_sub_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{module_name.split('.')[-1]}_{timestamp}.log"
    
    # Create Logger
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent adding multiple handlers if logger is called multiple times
    if not logger.handlers:
        # Formatter
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # File Handler (logs everything, including DEBUG)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console Handler (logs INFO and above to terminal)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger