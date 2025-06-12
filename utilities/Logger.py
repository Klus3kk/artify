"""
Logger System for Artify
Properly handles multiple loggers and file outputs
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class Logger:
    """
    Class that properly handles multiple loggers and file outputs
    """
    
    _configured_loggers = set()
    _root_configured = False
    
    @staticmethod
    def setup_logger(
        log_file: Optional[str] = None, 
        log_level: int = logging.INFO,
        logger_name: Optional[str] = None,
        format_string: Optional[str] = None
    ) -> logging.Logger:
        """
        Set up a logger with console and optional file logging.
        
        :param log_file: Name of the log file (optional).
        :param log_level: Logging level (default: INFO).
        :param logger_name: Name of the logger (default: based on log_file).
        :param format_string: Custom format string.
        :return: Configured logger instance.
        """
        
        # Ensure logs directory exists
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine logger name
        if logger_name is None:
            if log_file:
                logger_name = Path(log_file).stem
            else:
                logger_name = "Artify"
        
        # Get or create logger
        logger = logging.getLogger(logger_name)
        
        # Only configure if not already configured
        if logger_name not in Logger._configured_loggers:
            # Clear any existing handlers
            logger.handlers.clear()
            
            # Set level
            logger.setLevel(log_level)
            
            # Default format
            if format_string is None:
                format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            
            formatter = logging.Formatter(format_string)
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler (if specified)
            if log_file:
                # Determine full log file path
                if "/" in log_file or "\\" in log_file:
                    log_file_path = Path(log_file)
                else:
                    log_file_path = logs_dir / log_file
                
                # Ensure parent directory exists
                log_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create file handler
                file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
                file_handler.setLevel(log_level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
                logger.info(f"Logger '{logger_name}' configured with file output: {log_file_path}")
            else:
                logger.info(f"Logger '{logger_name}' configured with console output only")
            
            # Prevent propagation to root logger to avoid duplicate messages
            logger.propagate = False
            
            # Mark as configured
            Logger._configured_loggers.add(logger_name)
        
        return logger
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get an existing logger by name.
        
        :param name: Name of the logger.
        :return: Logger instance.
        """
        return logging.getLogger(name)
    
    @staticmethod
    def set_level_all(level: int):
        """
        Set logging level for all configured loggers.
        
        :param level: New logging level.
        """
        for logger_name in Logger._configured_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)
    
    @staticmethod
    def disable_console_output(logger_name: str):
        """
        Disable console output for a specific logger (keep only file output).
        
        :param logger_name: Name of the logger.
        """
        logger = logging.getLogger(logger_name)
        
        # Remove console handlers
        handlers_to_remove = []
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handlers_to_remove.append(handler)
        
        for handler in handlers_to_remove:
            logger.removeHandler(handler)
    
    @staticmethod
    def add_file_output(logger_name: str, log_file: str, level: int = logging.INFO):
        """
        Add file output to an existing logger.
        
        :param logger_name: Name of the logger.
        :param log_file: Path to the log file.
        :param level: Logging level for the file handler.
        """
        logger = logging.getLogger(logger_name)
        
        # Ensure logs directory exists
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine full log file path
        if "/" in log_file or "\\" in log_file:
            log_file_path = Path(log_file)
        else:
            log_file_path = logs_dir / log_file
        
        # Create file handler
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        
        # Use same formatter as existing handlers
        if logger.handlers:
            formatter = logger.handlers[0].formatter
        else:
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Added file output: {log_file_path}")
    
    @staticmethod
    def list_loggers():
        """
        List all configured loggers.
        
        :return: Set of logger names.
        """
        return Logger._configured_loggers.copy()
    
    @staticmethod
    def test_logger(logger_name: str = "test"):
        """
        Test logger functionality.
        
        :param logger_name: Name for the test logger.
        """
        # Test console only
        logger = Logger.setup_logger(logger_name=f"{logger_name}_console")
        logger.info("Test console message")
        logger.warning("Test console warning")
        logger.error("Test console error")
        
        # Test file output
        logger_file = Logger.setup_logger(
            log_file=f"{logger_name}_test.log", 
            logger_name=f"{logger_name}_file"
        )
        logger_file.info("Test file message")
        logger_file.warning("Test file warning")
        logger_file.error("Test file error")
        
        print(f"✓ Logger test complete. Check logs/{logger_name}_test.log")
        
        return logger, logger_file


# Legacy compatibility function
def setup_logger(log_file=None, log_level=logging.INFO):
    """
    Legacy compatibility function for existing code.
    """
    return Logger.setup_logger(log_file=log_file, log_level=log_level)

