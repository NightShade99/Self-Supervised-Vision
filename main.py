
import os
import argparse
import numpy as np
from datetime import datetime as dt
from models import (
    simclr, moco, byol, dino, pirl, barlow, simsiam, relic,
    deep_cluster, swav, sela
)

TASKS = ["train", "linear_eval", "get_features"]
NETWORKS = ["resnet18", "resnet50", "resnext50", "resnext101", "wide_resnet50", "wide_resnet101", "vit"]

ALGORITHMS = {
    "simclr": simclr.SimCLR,
    "moco": moco.MoCo,
    "byol": byol.BYOL,
    "dino": dino.DINO,
    "pirl": pirl.PIRL,
    "barlow": barlow.BarlowTwins,
    "simsiam": simsiam.SimSiam,
    "relic": relic.ReLIC,
    "deep_cluster": deep_cluster.DeepCluster,
    "swav": swav.SwAV,
    "sela": sela.SeLA
}

def _check_checkpoint_specified(args):
    if args["load"] is None:
        raise NotImplementedError("For inference tasks, model checkpoint must be specified using --load")
    else:
        pass


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, type=str, help="Path to configuration file")
    ap.add_argument("-m", "--arch", required=True, type=str, choices=NETWORKS, help="Encoder architecture to use")
    ap.add_argument("-a", "--algo", required=True, type=str, choices=ALGORITHMS, help="Self-supervised algorithm to work with")
    ap.add_argument("-t", "--task", required=True, type=str, choices=TASKS, help="Task to perform for chosen algorithm")
    ap.add_argument("-o", "--output", default=dt.now().strftime("%d-%m-%Y_%H-%M"), type=str, help="Path to output directory")
    ap.add_argument("-l", "--load", default=None, type=str, help="Path to directory containing trained checkpoints to be loaded")
    args = vars(ap.parse_args())

    # Initialize model based on algorithm
    model = ALGORITHMS[args["algo"]](args=args)
    task = args["task"]

    # Perform task
    if task == "train":
        model.train()

    elif task == "linear_eval":
        _check_checkpoint_specified(args)
        model.perform_linear_eval()

    elif task == "get_features":
        _check_checkpoint_specified(args)
        train_fvecs, train_gt = model.build_features(split="train")
        test_fvecs, test_gt = model.build_features(split="test")

        with open(os.path.join(model.output_dir, "train_fvecs.npy"), "w") as f:
            np.save(f, train_fvecs)
        with open(os.path.join(model.output_dir, "train_gt.npy"), "w") as f:
            np.save(f, train_gt)
        with open(os.path.join(model.output_dir, "test_fvecs.npy"), "w") as f:
            np.save(f, test_fvecs)
        with open(os.path.join(model.output_dir, "test_gt.npy"), "w") as f:
            np.save(f, test_gt)
