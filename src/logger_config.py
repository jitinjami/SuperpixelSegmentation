import logging
import os
from datetime import datetime

def setup_logger(log_dir='logs', log_file='superpixel_job.log'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(log_dir, f"{timestamp}_{log_file}")

    logging.basicConfig(
        filename=full_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('SuperpixelLogger')