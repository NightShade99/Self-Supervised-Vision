
import os
import math
import torch
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from networks import resnet, vit
from utils import common, losses
from utils import train_utils, data_utils, eval_utils

NETWORKS = {
    "resnet18": {"net": resnet.resnet18, "dim": 512},
    "resnet50": {"net": resnet.resnet50, "dim": 2048},
    "resnext50": {"net": resnet.resnext50_32x4d, "dim": 2048},
    "resnext101": {"net": resnet.resnext101_32x8d, "dim": 2048},
    "wide_resnet50": {"net": resnet.wide_resnet50_2, "dim": 2048},
    "wide_resnet101": {"net": resnet.wide_resnet101_2, "dim": 2048},
    "vit": {"net": vit.TransformerEncoder, "dim": None}
}


class EncoderModel(nn.Module):

    def __init__(self, encoder, encoder_dim, hidden_dim, projection_dim):
        super(EncoderModel, self).__init__()
        self.encoder = encoder
        self.proj_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim))

    def forward(self, x):
        x = self.encoder(x)
        x = self.proj_head(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class Prototypes(nn.Module):

    def __init__(self, hidden_dim, prototype_size):
        super(Prototypes, self).__init__()
        self.proto_size = prototype_size
        self.embedding = nn.Embedding(prototype_size, hidden_dim)

    def forward(self, device):
        indices = torch.arange(self.proto_size).long().to(device)
        vectors = F.normalize(self.embedding(indices), p=2, dim=-1)
        return vectors


class FeatureBank:
    """
    Since my batch sizes are small, features of previous batches
    will be stored here so that it may be used along with the current
    mini-batch's features, and meaningful partitioning into clusters
    is possible.
    """

    def __init__(self, bank_size, feature_dim):
        self.vectors = torch.FloatTensor(bank_size, feature_dim).zero_()
        self.bank_size = bank_size
        self.ptr = 0

    def add_vectors(self, fvecs):
        for i in range(fvecs.size(0)):
            self.vectors[self.ptr] = fvecs[i]
            self.ptr += 1
            if self.ptr >= self.bank_size:
                self.ptr = 0

    def return_vectors(self, device):
        vectors = self.vectors.to(device)
        return vectors


class SwAV:

    def __init__(self, args):
        assert args["arch"] in NETWORKS.keys(), f"Expected 'arch' to be one of {list(NETWORKS.keys())}"
        output_root = os.path.join("outputs/swav", args["arch"])

        self.config, self.output_dir, self.logger, self.device = common.initialize_experiment(args, output_root)
        self.train_loader, self.test_loader = data_utils.get_double_augment_dataloaders(**self.config["data"])
        run = wandb.init(**self.config["wandb"])
        self.logger.write("Wandb url: {}".format(run.get_url()), mode="info")

        encoder, encoder_dim = NETWORKS[args["arch"]].values()
        self.model = EncoderModel(encoder(**self.config["encoder"]), encoder_dim, self.config["hidden_dim"], self.config["proj_dim"]).to(self.device)
        self.prototypes = Prototypes(self.config["proj_dim"], self.config["prototype_size"]).to(self.device)
        self.feature_bank = FeatureBank(self.config["feature_bank_size"], self.config["proj_dim"])
        self.initialize_feature_bank()

        self.optim = train_utils.get_optimizer(self.config["optimizer"], params=list(self.model.parameters())+list(self.prototypes.parameters()))
        self.scheduler, self.warmup_epochs = train_utils.get_scheduler({**self.config["scheduler"], "epochs": self.config["epochs"]}, optimizer=self.optim)
        if self.warmup_epochs > 0:
            self.warmup_rate = (self.config["optimizer"]["lr"] - 1e-12) / self.warmup_epochs

        self.loss_fn = losses.SwavLoss(**self.config["loss_fn"])
        self.best_metric = 0

        if args["load"] is not None:
            self.load_checkpoint(args["load"])

    def save_checkpoint(self):
        state = {"encoder": self.model.state_dict()}
        torch.save(state, os.path.join(self.output_dir, "best_model.pt"))

    def load_checkpoint(self, ckpt_dir):
        if os.path.exists(os.path.join(ckpt_dir, "best_model.pt")):
            state = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=self.device)
            self.model.load_state_dict(state["encoder"])
            self.logger.print(f"Successfully loaded model from {ckpt_dir}")
        else:
            raise NotImplementedError(f"Could not find saved checkpoint at {ckpt_dir}")

    def adjust_learning_rate(self, epoch):
        if epoch <= self.warmup_epochs:
            for group in self.optim.param_groups:
                group["lr"] = 1e-12 + epoch * self.warmup_rate
        elif self.scheduler is not None:
            self.scheduler.step()
        else:
            pass

    def initialize_feature_bank(self):
        self.logger.print("Initializing feature bank", mode="info")
        fvecs, gt = self.build_features(split="train")
        fvecs = torch.tensor(fvecs[-self.config["feature_bank_size"]:]).float()
        self.feature_bank.add_vectors(fvecs)

    def train_step(self, batch):
        img_1, img_2 = batch["aug_1"].to(self.device), batch["aug_2"].to(self.device)
        z_1, z_2 = self.model(img_1), self.model(img_2)
        loss = self.loss_fn(z_1, z_2, self.prototypes(self.device), self.feature_bank.return_vectors(self.device))
        self.feature_bank.add_vectors(fvecs=torch.cat([z_1, z_2], 0).detach().cpu())

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {"loss": loss.item()}

    @torch.no_grad()
    def knn_validate(self):
        fvecs, gt = self.build_features(split="test")
        acc = eval_utils.compute_neighbor_accuracy(fvecs, gt)
        return acc

    @torch.no_grad()
    def build_features(self, split="train"):
        fvecs, gt = [], []

        if split == "train":
            for step, batch in enumerate(self.train_loader):
                img, trg = batch["img"].to(self.device), batch["label"].detach().cpu().numpy()
                z = self.model(img).detach().cpu().numpy()
                fvecs.append(z), gt.append(trg)
                common.progress_bar(progress=(step+1)/len(self.train_loader), desc="Building train features")
            print()

        elif split == "test":
            for step, batch in enumerate(self.test_loader):
                img, trg = batch["img"].to(self.device), batch["label"].detach().cpu().numpy()
                z = self.model(img).detach().cpu().numpy()
                fvecs.append(z), gt.append(trg)
                common.progress_bar(progress=(step+1)/len(self.test_loader), desc="Building test features")
            print()
        else:
            raise ValueError(f"Unrecognized split {split}, expected one of [train, test]")

        fvecs, gt = np.concatenate(fvecs, axis=0), np.concatenate(gt, axis=0)
        return fvecs, gt

    def perform_linear_eval(self):
        train_vecs, train_gt = self.build_features(split="train")
        test_vecs, test_gt = self.build_features(split="test")
        test_linear_acc = eval_utils.linear_evaluation(
            config = self.config["linear_eval"],
            train_data = {"fvecs": train_vecs, "labels": train_gt},
            test_data = {"fvecs": test_vecs, "labels": test_gt},
            num_classes = 10,
            device = self.device
        )
        self.logger.write("Test linear eval accuracy: {:.4f}".format(test_linear_acc), mode="info")

    def train(self):
        self.logger.print("Beginning training.", mode="info")
        for epoch in range(1, self.config["epochs"]+1):
            train_meter = common.AverageMeter()
            desc_str = "[TRAIN] Epoch {:4d}/{:4d}".format(epoch, self.config["epochs"])

            for step, batch in enumerate(self.train_loader):
                train_metrics = self.train_step(batch)
                wandb.log({"Train loss": train_metrics["loss"]})
                train_meter.add(train_metrics)
                common.progress_bar(progress=(step+1)/len(self.train_loader), desc=desc_str, status=train_meter.return_msg())
            print()
            self.logger.write("Epoch {:4d}/{:4d} ".format(epoch, self.config["epochs"]) + train_meter.return_msg(), mode="train")
            self.adjust_learning_rate(epoch)

            if epoch % self.config["eval_every"] == 0:
                knn_acc = self.knn_validate()
                self.logger.record("Epoch {:4d}/{:4d} [accuracy] {:.4f}".format(epoch, self.config["epochs"], knn_acc), mode="val")
                wandb.log({"KNN accuracy": knn_acc, "Epoch": epoch})

                if knn_acc > self.best_metric:
                    self.best_metric = knn_acc
                    self.save_checkpoint()
        print()
        self.logger.print("Completed training. Beginning linear evaluation.", mode="info")
        self.perform_linear_eval()
