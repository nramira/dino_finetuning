import logging
from datetime import datetime
from pathlib import Path

LOG_FILE: Path = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOGS_PATH: Path = Path.cwd() / "logs"
LOGS_PATH.mkdir(exist_ok=True)

LOG_FILE_PATH: Path = LOGS_PATH / LOG_FILE

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
