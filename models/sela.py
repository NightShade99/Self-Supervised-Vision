
import os
import math
import torch 
import wandb
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets
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

DATASETS = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100
}


class ClusteringModel(nn.Module):
    
    def __init__(self, backbone, feature_dim, num_clusters, num_cluster_heads):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone 
        self.num_heads = num_cluster_heads
        self.cluster_heads = nn.ModuleList([nn.Linear(feature_dim, num_clusters) for _ in range(num_cluster_heads)])
        
    def forward(self, inp):
        features = self.backbone(inp)
        logits = []
        for i in range(self.num_heads):
            logits.append(self.cluster_heads[i](features))
        return features, logits


class SeLA:
    
    def __init__(self, args):
        assert args["arch"] in NETWORKS.keys(), f"Expected 'arch' to be one of {list(NETWORKS.keys())}"
        output_root = os.path.join("outputs/sela", args["arch"])
        
        self.config, self.output_dir, self.logger, self.device = common.initialize_experiment(args, output_root)
        run = wandb.init(**self.config["wandb"])
        self.logger.write("Wandb url: {}".format(run.get_url()), mode="info")
        
        dataset_name, root = self.config["data"]["dataset_name"], self.config["data"]["root"]
        self.train_dset = DATASETS[dataset_name](root=root, train=True, transform=None, download=True)
        self.test_dset = DATASETS[dataset_name](root=root, train=False, transform=None, download=True)
        self.train_loader, self.test_loader = data_utils.get_pseudolabel_dataloaders(
            self.train_dset, self.test_dset, None, self.config["data"]["root"], self.config["data"]["transforms"], 
            self.config["data"]["batch_size"])
        
        encoder, encoder_dim = NETWORKS[args["arch"]]["net"](**self.config["encoder"]), NETWORKS[args["arch"]]["dim"]
        self.model = ClusteringModel(encoder, encoder_dim, self.config["num_clusters"], self.config["num_cluster_heads"]).to(self.device)
        self.optim = train_utils.get_optimizer(self.config["optimizer"], params=self.model.parameters())
        self.scheduler, self.warmup_epochs = train_utils.get_scheduler({"epochs": self.config["epochs"], **self.config["scheduler"]}, self.optim)
        if self.warmup_epochs > 0:
            self.warmup_rate = (self.config["optimizer"]["lr"] - 1e-12) / self.warmup_epochs 
        
        self.sl_epochs = [int(self.config["epochs"]*(i/(self.config["self_label_iters"]-1))**2) for i in range(1, self.config["self_label_iters"]-1)]
        self.lmbd = self.config["lambda"]
        self.alpha = torch.FloatTensor(self.config["num_clusters"], 1).normal_(0, 1).to(self.device)
        self.beta = torch.FloatTensor(self.config["data"]["batch_size"], 1).normal_(0, 1).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.best_metric = 0
        self.best_head = 0
        
        if args["load"] is not None:
            self.load_checkpoint(args["load"])
            
    def save_checkpoint(self):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "best_model.pt"))

    def load_checkpoint(self, ckpt_dir):
        if os.path.exists(os.path.join(ckpt_dir, "best_model.pt")):
            state = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=self.device)
            self.model.load_state_dict(state)
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
    def knn_validate(self):
        fvecs, gt = self.build_features(split="test")
        acc = eval_utils.compute_neighbor_accuracy(fvecs, gt)
        return acc

    @torch.no_grad()
    def build_features(self, split="train", desc="Building features"):
        fvecs, gt = [], []

        if split == "train":
            for step, batch in enumerate(self.train_loader):
                img, trg = batch["img"].to(self.device), batch["label"].detach().cpu().numpy()
                features, _ = self.model(img)
                fvecs.append(features.detach().cpu().numpy()), gt.append(trg)
                common.progress_bar(progress=(step+1)/len(self.train_loader), desc=desc)
            print()

        elif split == "test":
            for step, batch in enumerate(self.test_loader):
                img, trg = batch["img"].to(self.device), batch["label"].detach().cpu().numpy()
                features, _ = self.model(img)
                fvecs.append(features.detach().cpu().numpy()), gt.append(trg)
                common.progress_bar(progress=(step+1)/len(self.test_loader), desc=desc)
            print()
        else:
            raise ValueError(
                f"Unrecognized split {split}, expected one of [train, test]")

        fvecs, gt = np.concatenate(fvecs, axis=0), np.concatenate(gt, axis=0)
        return fvecs, gt
        
    def train_step(self, batch):
        imgs, labels = batch["aug"].to(self.device), batch["label"].to(self.device)
        _, logits = self.model(imgs)
        losses = []
        for i in range(self.config["num_cluster_heads"]):
            losses.append(self.loss_fn(logits[i], labels))
        loss = sum(losses)
        self.best_head = torch.tensor(losses).argmin().item()
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {"loss": loss.item()}
    
    @torch.no_grad()
    def self_label_step(self, num_iters=80):
        pseudo_labels = []
        for step, batch in enumerate(self.train_loader):
            imgs = batch["img"].to(self.device)
            _, logits = self.model(imgs)
            log_probs = torch.pow(F.log_softmax(logits[self.best_head], -1), self.lmbd).t()
            
            for _ in range(num_iters):
                self.alpha = 1.0 / torch.mm(log_probs, self.beta)
                self.beta = 1.0 / torch.mm(self.alpha.t(), log_probs).t()
        
            alpha_diag = torch.eye(self.alpha.size(0)).to(self.device) * self.alpha 
            beta_diag = torch.eye(self.beta.size(0)).to(self.device) * self.beta 
            pseudo_labels.extend((alpha_diag @ log_probs @ beta_diag).t().argmax(-1).detach().cpu().numpy().tolist())
            common.progress_bar(progress=(step+1)/len(self.train_loader), desc="Self-labelling", status="")
        print("\n")
        self.train_loader, self.test_loader = data_utils.get_pseudolabel_dataloaders(
            self.train_dset, self.test_dset, pseudo_labels, self.config["data"]["root"], self.config["data"]["transforms"], 
            self.config["data"]["batch_size"]
        )
        
    def perform_linear_eval(self):
        self.train_loader, self.test_loader = data_utils.get_pseudolabel_dataloaders(
            self.train_dset, self.test_dset, None, self.config["data"]["root"], self.config["data"]["transforms"], 
            self.config["data"]["batch_size"]
        )
        train_vecs, _, train_gt = self.build_features(split="train")
        test_vecs, _, test_gt = self.build_features(split="test")
        test_linear_acc = eval_utils.linear_evaluation(
            config = self.config["linear_eval"],
            train_data = {"fvecs": train_vecs, "labels": train_gt},
            test_data = {"fvecs": test_vecs, "labels": test_gt},
            num_classes = 10,
            device = self.device
        )
        self.logger.write("Test linear eval accuracy: {:.4f}".format(test_linear_acc), mode="info")
        
    def train(self):
        # Begin by generate some labels using self-labelling
        self.self_label_step(num_iters=self.config["self_label_iters"])
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
            
            if epoch in self.sl_epochs:
                self.self_label_step(num_iters=self.config["self_label_iters"])

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