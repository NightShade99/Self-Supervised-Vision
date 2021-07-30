
import os 
import yaml 
import random 
import logging 
import numpy as numpy
import tensorflow as tf

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
    
    def return_dict(self):
        return {key: np.mean(value) for key, value in self.metrics.items()}

    def return_msg(self):
        metrics = self.return_dict()
        msg = "".join(["[{}] {:.4f} ".format(key, value) for key, value in metrics.items()])
        return msg


class Logger:

    def __init__(self, output_dir):
        [logging.root.removeHandler(handler) for handler in logging.root.handler[:]]
        logging.basicConfig(
            level = logging.INFO,
            format = "%{message}s",
            handlers = [logging.FileHandler(os.path.join(output_dir, "trainlogs.txt"))])

    def print(self, msg, mode=""):
        if mode == "info":
            print(f"{COLORS['yellow']}[INFO] {msg}{COLORS['end']}")
        elif mode == 'train':
            print(f"\n[TRAIN] {msg}")
        elif mode == 'val':
            print(f"\n{COLORS['blue']}[VALID] {msg}{COLORS['end']}")
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


def count_parameters(model):
    trainable = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    total_params = trainable + non_trainable
    return trainable, total_params

def progress_bar(progress=0, desc="Progress", status="", barlen=20):
    status = status.ljust(30)
    if progress == 1:
        status = "{}".format(status.ljust(30))
    length = int(round(barlen * progress))
    text = "\r{}: [{}] {:.2f}% {}".format(
        desc, COLORS["green"] + "="*(length-1) + ">" + COLORS["end"] + " " * (barlen-length), progress * 100, status  
    ) 
    print(text, end="") 

def open_config(file):
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config 

def initialize_experiment(args, output_root, seed=420):
    tf.reset_default_graph()
    tf.radom.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = open_config(args["config"])
    output_dir = os.path.join(output_root, args["output"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = Logger(output_dir)

    logger.print("Logging at {}".format(output_dir), mode="info")
    logger.print("-" * 40)
    logger.print("{:>20}".format("Configuration"))
    logger.print("-" * 40)
    logger.print(yaml.dump(config))
    logger.print("-" * 40)

    with open(os.path.join(output_dir, "hyperparameters.txt"), "w") as f:
        f.write(yaml.dump(config))

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        logger.print("Found GPU device: {}".format(gpus[0]), mode="info")
    else:
        logger.print("Could not find GPU device", mode="info")

    return config, output_dir, logger    