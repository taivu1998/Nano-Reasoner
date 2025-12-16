import os
import random
import numpy as np
import torch
import logging
import glob
import re

def seed_everything(seed: int = 42):
    """Sets the seed for reproducibility across all libraries."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(name: str, log_dir: str = None):
    """Configures a standardized logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    
    return logger

def find_latest_checkpoint(checkpoint_dir: str):
    """Find the latest checkpoint in a directory.

    Returns:
        Tuple of (checkpoint_path, step_number) or (None, 0) if no checkpoint found.
    """
    if not os.path.exists(checkpoint_dir):
        return None, 0

    checkpoints = glob.glob(os.path.join(checkpoint_dir, "step_*"))
    if not checkpoints:
        return None, 0

    step_pattern = re.compile(r'step_(\d+)')
    steps = []
    for ckpt in checkpoints:
        match = step_pattern.search(ckpt)
        if match:
            steps.append((int(match.group(1)), ckpt))

    if not steps:
        return None, 0

    steps.sort(reverse=True)
    return steps[0][1], steps[0][0]