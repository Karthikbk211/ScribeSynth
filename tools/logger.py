import logging
import os


def set_log(output_dir, cfg_file, log_name):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f'{log_name}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    logger.info(f'config file: {cfg_file}')
    logger.info(f'log file: {log_path}')
    return logger
