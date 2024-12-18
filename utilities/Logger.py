import logging


class Logger:
    @staticmethod
    def setup_logger(log_file=None, log_level=logging.INFO):
        """
        Set up the logger with console and optional file logging.

        :param log_file: Path to the log file (optional).
        :param log_level: Logging level (default: INFO).
        :return: Configured logger instance.
        """
        handlers = [logging.StreamHandler()]
        if log_file:
            handlers.append(logging.FileHandler(log_file))

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=handlers,
        )
        return logging.getLogger("Artify")
