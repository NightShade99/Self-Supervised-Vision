
import os
import jax
import yaml
import random
import logging
import numpy as np

COLORS = {
    "yellow": "\x1b[33m",
    "blue": "\x1b[94m",
    "green": "\x1b[32m",
    "red": "\x1b[33m",
    "end": "\033[0m"
}


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}

    def add(self, metrics: dict):
        if len(self.metrics) == 0:
            self.metrics = {key: [value] for key, value in metrics.items()}
        else:
            for key, value in metrics.items():
                if key in self.metrics.keys():
                    self.metrics[key].append(value)
                else:
                    self.metrics[key] = [value]

    def avg(self):
        return {key: np.mean(value) for key, value in self.metrics.items()}

    def msg(self):
        metrics = self.avg()
        msg = " ".join(["[{}] {:.4f}".format(name, value) for name, value in metrics.items()])
        return msg


class Logger:
    def __init__(self, output_dir):
        [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
        logging.basicConfig(
            level = logging.INFO,
            format = "%(message)s",
            handlers = [logging.FileHandler(os.path.join(output_dir, "logs.txt"))])

    def print(self, msg, mode=""):
        if mode == "info":
            print(f"{COLORS['yellow']}[INFO]{COLORS['end']} {msg}")
        elif mode == 'train':
            print(f"{COLORS['green']}[TRAIN]{COLORS['end']} {msg}")
        elif mode == 'val':
            print(f"{COLORS['blue']}[VALID]{COLORS['end']} {msg}")
        else:
            print(f"{msg}")

    def write(self, msg, mode):
        if mode == "info":
            msg = f"[INFO] {msg}"
        elif mode == "train":
            msg = f"[TRAIN] {msg}"
        elif mode == "val":
            msg = f"[VALID] {msg}"
        logging.info(msg)

    def record(self, msg, mode):
        self.print(msg, mode)
        self.write(msg, mode)


def pbar(progress=0, desc="Progress", barlen=20, status=''):
    status = status.ljust(30)
    if progress == 1:
        status = "{}".format(status.ljust(30))
    length = int(round(barlen * progress))
    text = "\r{}: [{}] {:.2f}% {}".format(
        desc, 
        COLORS["green"] + "="*(length-1) + ">" + COLORS["end"] + "-" * (barlen-length), 
        progress * 100, 
        status
    )
    print(text, end="" if progress < 1 else "\n")