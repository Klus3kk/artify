import logging
from pathlib import Path


class Logger:
    @staticmethod
    def setup_logger(log_file=None, log_level=logging.INFO):
        """
        Set up the logger with console and optional file logging.

        :param log_file: Name of the log file (optional).
        :param log_level: Logging level (default: INFO).
        :return: Configured logger instance.
        """
        # Ensure logs directory exists
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)

        handlers = [logging.StreamHandler()]
        if log_file:
            # If `log_file` is a path, use it as is. Otherwise, prepend with `logs/`.
            log_file_path = Path(log_file) if "/" in log_file else logs_dir / log_file
            handlers.append(logging.FileHandler(log_file_path))

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=handlers,
        )
        return logging.getLogger(log_file or "Artify")
