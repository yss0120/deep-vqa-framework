# src/utils/logging_utils.py
import atexit
import sys
import time
from functools import wraps
from pathlib import Path

from loguru import logger


def _csv_safe_patcher(record):
    """
    Escape newline characters and double quotes in log messages to avoid CSV format corruption.
    """
    msg = record["message"]
    # 1. Avoid double quotes: Replace " with "" (CSV official escaping specification)
    msg = msg.replace('"', '""')
    # 2. Avoid newline characters: Replace newline characters with the special visible character \\n to keep them on a single line.
    if "\n" in msg:
        msg = msg.replace("\n", "\\n")

    # Re-inject into the record (create a clean field specifically for CSV consumption, without polluting the console).
    record["extra"]["csv_message"] = msg


def on_exit():
    # The logger can still log normally even if the program exits abnormally.
    if sys.exc_info()[0]:
        logger.error(f"⚠️ [System] The training pipeline terminated abnormally: {sys.exc_info()[1]}")
    else:
        logger.info("✅ [System] The training pipeline has been successfully completed.")


def log_prepare(model_name: str = "resnet50", dataset_name: str = "TID2013"):
    """
    Configure global logging: console output at INFO level, file output at DEBUG level, and CSV logs include escaped messages.
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    # Forcefully clear the Loguru default global processor (prevent DEBUG screen refresh and freeze tqdm).
    logger.remove()

    # Dynamically configure the global cleaner (Patcher) to automatically obtain CSV safe fields from logs.
    logger.configure(patcher=_csv_safe_patcher)

    # Configure the console output: simplified, high-definition, and non-disruptive (tqdm)
    console_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <5}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>"
    logger.add(sys.stderr, level="INFO", format=console_format)

    log_dir = PROJECT_ROOT / "results" / "train_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Time format: 年月日_时分秒 (e.g., 20260519_143025)
    current_timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_filename = f"{current_timestamp}_{model_name.lower()}_{dataset_name.lower()}"

    # Redirect the output stream to a standard .log - retain full debug details
    file_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <5} | {name}:{function}:{line} - {message}"
    logger.add(log_dir / f"{base_filename}.log", rotation="500 MB", level="DEBUG", format=file_format, encoding="utf-8")

    # Formatted output to CSV log
    csv_format = '{time:YYYY-MM-DD HH:mm:ss.SSS},{level},{name},{function},{line},"{extra[csv_message]}"'
    csv_header = "timestamp,level,module,function,line,message\n"
    csv_path = log_dir / f"{base_filename}.csv"

    # Define the rotation callback and write it to the CSV header
    def csv_rotation_callback(message, file_object):
        """Loguru 轮转新日志文件时自动触发，为其续上标准 CSV 表头"""
        file_object.write(csv_header)

    # Write CSV header on first run
    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_header)

    # Add CSV log output
    logger.add(csv_path, rotation=csv_rotation_callback, level="DEBUG", format=csv_format, encoding="utf-8")

    logger.info("⚙️  [System] Log system initialization complete.")
    logger.info(f"📁 [Outputs] TXT text log: results/logs/{base_filename}.log")
    logger.info(f"📁 [Outputs] CSV chart log: results/logs/{base_filename}.csv")

    atexit.register(on_exit)
    return base_filename


def time_it(func):
    """Execution time of statistical functions"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        module_name = func.__module__.split(".")[-1]
        logger.info(f"⏱️  [Timer] [{module_name}] Execution time for {func.__name__}: {end_time - start_time:.4f} seconds")
        return result

    return wrapper
