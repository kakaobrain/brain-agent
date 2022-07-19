import logging
from colorlog import ColoredFormatter

log = logging.getLogger('rl')
log.setLevel(logging.DEBUG)
log.handlers = []
log.propagate = False
log_level = logging.DEBUG

stream_handler = logging.StreamHandler()
stream_handler.setLevel(log_level)

stream_formatter = ColoredFormatter(
    '%(log_color)s[%(asctime)s][%(process)05d] %(message)s',
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'white,bold',
        'INFOV': 'cyan,bold',
        'WARNING': 'yellow',
        'ERROR': 'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
stream_handler.setFormatter(stream_formatter)
log.addHandler(stream_handler)

def init_logger(log_level='debug', file_path=None):
    log.setLevel(logging.getLevelName(str.upper(log_level)))
    if file_path is not None:
        file_handler = logging.FileHandler(file_path)
        file_formatter = logging.Formatter(fmt='[%(asctime)s][%(process)05d] %(message)s', datefmt=None, style='%')
        file_handler.setFormatter(file_formatter)
        log.addHandler(file_handler)
    for h in log.handlers:
        h.setLevel(logging.getLevelName(str.upper(log_level)))


