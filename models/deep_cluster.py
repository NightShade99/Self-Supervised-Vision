
import os
import math
import torch 
import wandb
import faiss
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

class ProjectionHead(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def __call__(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return x


class DeepCluster:

    def __init__(self, args):
        assert args["arch"] in NETWORKS.keys(), f"Expected 'arch' to be one of {list(NETWORKS.keys())}"
        output_root = os.path.join("outputs/deep_cluster", args["arch"])
        
        self.config, self.output_dir, self.logger, self.device = common.initialize_experiment(args, output_root)
        self.train_loader, self.test_loader = data_utils.get_double_augment_dataloaders(**self.config["data"])
        self.pseudolabel_loader = None
        self.num_classes = self.config["num_classes"]
        run = wandb.init(**self.config["wandb"])
        self.logger.write("Wandb url: {}".format(run.get_url()), mode="info")

        encoder, encoder_dim = NETWORKS[args["arch"]].values()
        self.encoder = encoder(**self.config["encoder"]).to(self.device)
        self.clf_head = nn.Linear(encoder_dim, self.num_classes).to(self.device)
        self.optim = train_utils.get_optimizer(self.config["optimizer"], params=list(self.encoder.parameters())+list(self.clf_head.parameters()))
        self.scheduler, self.warmup_epochs = train_utils.get_scheduler({**self.config["scheduler"], "epochs": self.config["epochs"]}, optimizer=self.optim)
        if self.warmup_epochs > 0:
            self.warmup_rate = (self.config["optimizer"]["lr"] - 1e-12) / self.warmup_epochs

        self.loss_fn = nn.CrossEntropyLoss()
        self.best_metric = 0
        
        if args["load"] is not None:
            self.load_checkpoint(args["load"])

    def save_checkpoint(self):
        state = {"encoder": self.encoder.state_dict(), "clf_head": self.clf_head.state_dict()}
        torch.save(state, os.path.join(self.output_dir, "best_model.pt"))

    def load_checkpoint(self, ckpt_dir):
        if os.path.exists(os.path.join(ckpt_dir, "encoder")):
            state = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=self.device)
            self.encoder.load_state_dict(state["encoder"])
            self.clf_head.load_state_dict(state["clf_head"])
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

    def train_step(self, batch):
        imgs, labels = batch["img"].to(self.device), batch["label"].to(self.device)
        features = F.normalize(self.encoder(imgs), p=2, dim=-1)
        output = self.clf_head(features)
        loss = self.loss_fn(output, labels)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()        
        return {"loss": loss.item()}

    @torch.no_grad()
    def generate_pseudolabels(self):
        images, features, pseudo_labels = [], [], []
        for step, batch in enumerate(self.train_loader):
            imgs = batch["img"].to(self.device)
            fvecs = F.normalize(self.encoder(imgs), p=2, dim=-1)
            preds = F.softmax(self.clf_head(fvecs), -1).argmax(-1)
            features.append(fvecs), images.append(imgs), pseudo_labels.append(preds)
            common.progress_bar(progress=(step+1)/len(self.train_loader), desc="Generating pseudolabels", status="")
        print()
        images, features = torch.cat(images, 0).detach().cpu(), torch.cat(features, 0).detach().cpu().numpy()
        kmeans = faiss.Kmeans(d=features.shape[1], k=self.num_classes, niter=300, nredo=10)
        kmeans.train(features.astype(np.float32))

        cluster_idx = kmeans.index.search(features.astype(np.float32), 1)[1].reshape(-1,)
        pseudo_labels = torch.cat(pseudo_labels, 0).detach().cpu().numpy()
        cluster_label_map = eval_utils.hungarian_match(cluster_idx, pseudo_labels, pred_k=self.num_classes, targets_k=self.num_classes)
        final_labels = torch.tensor([cluster_label_map[c] for c in cluster_idx]).long()
        self.pseudolabel_loader = data_utils.get_pseudolabel_dataloaders(images, final_labels, self.config["data"]["batch_size"])

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
                z = self.encoder(img)
                z = F.normalize(z, dim=-1, p=2).detach().cpu().numpy()
                fvecs.append(z), gt.append(trg)
                common.progress_bar(progress=(step+1)/len(self.train_loader), desc="Building train features")
            print()

        elif split == "test":
            for step, batch in enumerate(self.test_loader):
                img, trg = batch["img"].to(self.device), batch["label"].detach().cpu().numpy()
                z = self.encoder(img)
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
            num_classes = self.config["num_classes"],
            device = self.device
        )
        self.logger.write("Test linear eval accuracy: {:.4f}".format(test_linear_acc), mode="info")

    def train(self):
        self.logger.print("Beginning training.", mode="info")
        for epoch in range(1, self.config["epochs"]+1):
            train_meter = common.AverageMeter()
            desc_str = "[TRAIN] Epoch {:4d}/{:4d}".format(epoch, self.config["epochs"])

            self.generate_pseudolabels()
            for step, batch in enumerate(self.pseudolabel_loader):
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