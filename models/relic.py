
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
    "wide_resnet101": {"net": resnet.wide_resnet101_2, "dim": 2048}
}


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.bn1(self.fc1(x))))


class OnlineNetwork(nn.Module):
    
    def __init__(self, encoder, encoder_dim, projection_dim):
        super(OnlineNetwork, self).__init__()
        self.encoder = encoder 
        self.proj_head = MLP(encoder_dim, projection_dim)
        self.pred_head = MLP(projection_dim, projection_dim)

    def forward(self, x):
        x = self.pred_head(self.proj_head(self.encoder(x)))
        return F.normalize(x, dim=-1, p=2)


class TargetNetwork(nn.Module):

    def __init__(self, encoder, encoder_dim, projection_dim):
        super(TargetNetwork, self).__init__()
        self.encoder = encoder 
        self.proj_head = MLP(encoder_dim, projection_dim)

    def forward(self, x):
        x = self.proj_head(self.encoder(x))
        return F.normalize(x, dim=-1, p=2)


class InvariantCausalRepresentationLearning:

    def __init__(self, args):
        assert args["arch"] in NETWORKS.keys(), f"Expected 'arch' to be one of {list(NETWORKS.keys())}"
        output_root = os.path.join("outputs/byol", args["arch"])
        
        self.config, self.output_dir, self.logger, self.device = common.initialize_experiment(args, output_root)
        self.train_loader, self.test_loader = data_utils.get_double_augment_dataloaders(**self.config["data"])
        run = wandb.init(**self.config["wandb"])
        self.logger.write("Wandb url: {}".format(run.get_url()), mode="info")

        encoder, encoder_dim = NETWORKS[args["arch"]].values()
        self.online_network = OnlineNetwork(encoder(**self.config["encoder"]), encoder_dim, self.config["proj_dim"]).to(self.device)
        self.target_network = TargetNetwork(encoder(**self.config["encoder"]), encoder_dim, self.config["proj_dim"]).to(self.device)
        self.max_steps = self.config["epochs"] * len(self.train_loader)
        self.tau = self.config.get("tau", 0.996)
 
        for p in self.target_network.parameters():
            p.requires_grad = False

        self.optim = train_utils.get_optimizer(self.config["optimizer"], params=self.online_network.parameters())
        self.scheduler, self.warmup_epochs = train_utils.get_scheduler({**self.config["scheduler"], "epochs": self.config["epochs"]}, optimizer=self.optim)
        if self.warmup_epochs > 0:
            self.warmup_rate = (self.config["optimizer"]["lr"] - 1e-12) / self.warmup_epochs

        self.loss_fn = losses.RelicLoss(**self.config["loss_fn"])
        self.best_metric = 0
        self.start_epoch = 1
        
        if args["load"] is not None:
            self.load_checkpoint(args["load"])

        if args["resume"] is not None:
            self.load_state(args["resume"])

    def save_state(self, epoch):
        state = {
            "epoch": epoch+1,
            "online": self.online_network.state_dict(),
            "target": self.target_network.state_dict(),
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None
        }
        torch.save(state, os.path.join(self.output_dir, "last_state.pt"))

    def load_state(self, model_dir):
        location = os.path.join(model_dir, "last_state.pt")
        if os.path.exists(location):
            state = torch.load(location, map_location=self.device)
            self.online_network.load_state_dict(state["online"])
            self.target_network.load_state_dict(state["target"])
            self.optim.load_state_dict(state["optim"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state["scheduler"])
            self.start_epoch = state["epoch"]
            self.output_dir = model_dir 
            self.logger.print("Successfully load saved state", mode="info")
        else:
            raise ValueError(f"Could not find last_state.pt at {model_dir}")

    def save_checkpoint(self):
        state = {"encoder": self.online_network.state_dict()}
        torch.save(state, os.path.join(self.output_dir, "best_model.pt"))

    def load_checkpoint(self, ckpt_dir):
        if os.path.exists(os.path.join(ckpt_dir, "encoder")):
            state = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=self.device)
            self.online_network.load_state_dict(state["encoder"])
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

    def update_tau(self, step):
        tau_upper, tau_lower = self.config.get("tau_upper", 1.0), self.config.get("tau_lower", 0.996)
        self.tau = tau_upper - (tau_upper - tau_lower) * (math.cos(math.pi * step / self.max_steps) + 1) / 2

    @torch.no_grad()
    def momentum_update(self):
        for o_param, t_param in zip(self.online_network.parameters(), self.target_network.parameters()):
            t_param.data = self.tau * t_param.data + (1.0 - self.tau) * o_param.data

    def train_step(self, batch):
        img_orig, img_1, img_2 = batch["img"].to(self.device), batch["aug_1"].to(self.device), batch["aug_2"].to(self.device)
        online_1, target_1 = self.online_network(img_1), self.target_network(img_1)
        online_2, target_2 = self.online_network(img_2), self.target_network(img_2)
        orig_features = self.online_network(img_orig)
        loss_1, loss_2 = self.loss_fn(online_1, target_2, orig_features), self.loss_fn(online_2, target_1, orig_features)
        loss = loss_1 + loss_2

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
                z = self.online_network(img).detach().cpu().numpy()
                fvecs.append(z), gt.append(trg)
                common.progress_bar(progress=(step+1)/len(self.train_loader), desc="Building train features")
            print()

        elif split == "test":
            for step, batch in enumerate(self.test_loader):
                img, trg = batch["img"].to(self.device), batch["label"].detach().cpu().numpy()
                z = self.online_network(img).detach().cpu().numpy()
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
        for epoch in range(self.start_epoch, self.config["epochs"]+1):
            train_meter = common.AverageMeter()
            desc_str = "[TRAIN] Epoch {:4d}/{:4d}".format(epoch, self.config["epochs"])

            for step, batch in enumerate(self.train_loader):
                train_metrics = self.train_step(batch)
                wandb.log({"Train loss": train_metrics["loss"]})
                train_meter.add(train_metrics)
                common.progress_bar(progress=(step+1)/len(self.train_loader), desc=desc_str, status=train_meter.return_msg())
                self.update_tau(step)
                self.momentum_update()
            print()
            self.logger.write("Epoch {:4d}/{:4d} ".format(epoch, self.config["epochs"]) + train_meter.return_msg(), mode="train")
            self.adjust_learning_rate(epoch)
            self.save_state(epoch)

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
