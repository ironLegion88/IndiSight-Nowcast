import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

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
    base_log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_dir = base_log_dir / log_sub_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # Use stable per-module filenames and rotate them.
    log_file = log_dir / f"{module_name.split('.')[-1]}.log"
    
    # Create Logger
    logger = logging.getLogger(module_name)
    requested_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, requested_level, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    
    # Prevent adding multiple handlers if logger is called multiple times
    if not logger.handlers:
        # Formatter
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Rotating file handler for long-running production processes.
        max_bytes = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))
        backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8',
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # Console Handler (logs INFO and above to terminal)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger