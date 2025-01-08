# Logging in Artify

Logging is an essential component of Artify, providing insights into the system's internal workings, error handling, and debugging.

## Overview

Artify uses a centralized logging system to log activities across:
1. **CLIHandler** (Command-Line Interface)
2. **UIHandler** (Streamlit Interface)
3. **Core Components**:
   - StyleTransferModel
   - HuggingFaceHandler
   - ImageProcessor
   - StyleRegistry

Logs are stored in log files specific to each module for better traceability and debugging.

## Logging Configuration

The `Logger` utility in `utilities/Logger.py` configures and manages loggers for the entire project.

### Logger Class
```python
import logging

class Logger:
    @staticmethod
    def setup_logger(log_file, log_level=logging.INFO):
        logger = logging.getLogger()
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.setLevel(log_level)
        logger.addHandler(handler)
        return logger
```

### Features
- Logs are timestamped for tracking.
- Levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
- Separate log files for:
  - CLI (`cli.log`)
  - UI (`artify_ui.log`)
  - Hugging Face (`huggingface.log`)

## Examples

### CLI Logs
Logs generated during a CLI run are saved in `cli.log`. Example:
```plaintext
2025-01-08 23:30:02,900 - INFO - [1/5] Validating input arguments...
2025-01-08 23:30:02,901 - INFO - [3/5] Ensuring model for 'impressionism'...
2025-01-08 23:30:02,901 - INFO - Model for 'impressionism' not found locally. Downloading...
2025-01-08 23:30:03,211 - INFO - Model for 'impressionism' downloaded and saved to models/impressionism_model.pth.
```

### UI Logs
Logs generated during UI actions are saved in `artify_ui.log`. Example:
```plaintext
2025-01-08 23:33:18,456 - INFO - Style selected: Impressionism
2025-01-08 23:33:19,123 - INFO - Content image uploaded: sample.jpg
2025-01-08 23:33:21,456 - ERROR - Failed to apply style transfer: Model file missing.
```

### Hugging Face Logs
Logs for Hugging Face operations are stored in `huggingface.log`. Example:
```plaintext
2025-01-08 23:30:03,211 - INFO - Downloading 'impressionism_model.pth' from Hugging Face repository 'ClueSec/artify-models'...
2025-01-08 23:30:03,300 - ERROR - Error downloading 'impressionism_model.pth': Model file not found in repository.
```

## Best Practices

1. **Keep Logs Modular**:
   - Separate log files for different modules reduce clutter and improve traceability.

2. **Error Reporting**:
   - Always log exceptions with stack traces for debugging.
   - Example:
     ```python
     try:
         # Operation
     except Exception as e:
         logger.error(f"Operation failed: {e}", exc_info=True)
     ```

3. **Log Cleanup**:
   - Archive old logs periodically to avoid clutter.
   - Example:
     - Use a script or log rotation tools to manage logs efficiently.

4. **Real-Time Monitoring**:
   - Use tools like `tail` or log viewers for real-time debugging:
     ```bash
     tail -f cli.log
     ```
