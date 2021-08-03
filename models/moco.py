
import os
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
    "wide_resnet101": {"net": resnet.wide_resnet101_2, "dim": 2048}
}


class MemoryBank:

    def __init__(self, queue_size, feature_size):
        self.bank = torch.FloatTensor(queue_size, feature_size).zero_()
        self.bank = F.normalize(self.bank, dim=-1, p=2)
        self.size = queue_size
        self.ptr = 0 
        
    def add_batch(self, batch):
        for row in batch:
            self.bank[self.ptr] = F.normalize(row, dim=-1, p=2) 
            self.ptr += 1
            if self.ptr >= self.size:
                self.ptr = 0

    def get_vectors(self):
        return self.bank


class EncoderModel(nn.Module):
    """ This simply adds a projection head (MLP) to the backbone """
    
    def __init__(self, encoder, encoder_dim):
        super(EncoderModel, self).__init__()
        self.encoder = encoder 
        self.relu = nn.ReLU() 
        self.proj_head = nn.Linear(encoder_dim, encoder_dim)

    def forward(self, x):
        return self.proj_head(self.relu(self.encoder(x)))


class MomentumContrast:

    def __init__(self, args):
        assert args["arch"] in NETWORKS.keys(), f"Expected 'arch' to be one of {list(NETWORKS.keys())}"
        output_root = os.path.join("outputs/moco", args["arch"])
        
        self.config, self.output_dir, self.logger, self.device = common.initialize_experiment(args, output_root)
        self.train_loader, self.test_loader = data_utils.get_double_augment_dataloaders(**self.config["data"])
        run = wandb.init(**self.config["wandb"])
        self.logger.write("Wandb url: {}".format(run.get_url()), mode="info")

        encoder, encoder_dim = NETWORKS[args["arch"]].values()
        self.query_encoder = EncoderModel(encoder=encoder(**self.config["encoder"]), encoder_dim=encoder_dim).to(self.device)
        self.key_encoder = EncoderModel(encoder=encoder(**self.config["encoder"]), encoder_dim=encoder_dim).to(self.device)
        self.memory_bank = MemoryBank(self.config["queue_size"], encoder_dim)
        self.m = self.config.get("momentum", 0.999)

        self.key_encoder.load_state_dict(self.query_encoder.state_dict())
        for p in self.key_encoder.parameters():
            p.requires_grad = False

        self.optim = train_utils.get_optimizer(self.config["optimizer"], params=self.query_encoder.parameters())
        self.scheduler, self.warmup_epochs = train_utils.get_scheduler({**self.config["scheduler"], "epochs": self.config["epochs"]}, optimizer=self.optim)
        if self.warmup_epochs > 0:
            self.warmup_rate = (self.config["optimizer"]["lr"] - 1e-12) / self.warmup_epochs

        self.loss_fn = losses.InfoNCELoss(**self.config["loss_fn"])
        self.best_metric = 0
        
        if args["load"] is not None:
            self.load_checkpoint(args["load"])

    def save_checkpoint(self):
        state = {"encoder": self.query_encoder.state_dict()}
        torch.save(state, os.path.join(self.output_dir, "best_model.pt"))

    def load_checkpoint(self, ckpt_dir):
        if os.path.exists(os.path.join(ckpt_dir, "encoder")):
            state = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=self.device)
            self.query_encoder.load_state_dict(state["encoder"])
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

    @torch.no_grad()
    def momentum_update(self):
        for q_param, k_param in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            k_param.data = self.m * k_param.data + (1.0 - self.m) * q_param.data

    def train_step(self, batch):
        img_1, img_2 = batch["aug_1"].to(self.device), batch["aug_2"].to(self.device)
        query = self.query_encoder(img_1)
        keys = self.key_encoder(img_2)
        loss = self.loss_fn(query, keys, self.memory_bank.get_vectors().to(self.device))

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()        
        
        self.momentum_update()
        self.memory_bank.add_batch(keys)
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
                z = self.query_encoder(img)
                z = F.normalize(z, dim=-1, p=2).detach().cpu().numpy()
                fvecs.append(z), gt.append(trg)
                common.progress_bar(progress=(step+1)/len(self.train_loader), desc="Building train features")
            print()

        elif split == "test":
            for step, batch in enumerate(self.test_loader):
                img, trg = batch["img"].to(self.device), batch["label"].detach().cpu().numpy()
                z = self.query_encoder(img)
                z = F.normalize(z, dim=-1, p=2).detach().cpu().numpy()
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
            self.logger.write("[TRAIN] Epoch {:4d}/{:4d} ".format(epoch, self.config["epochs"]) + train_meter.return_msg(), mode="train")
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
