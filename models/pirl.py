
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

    def __init__(self, data_size, feature_size, momentum=0.5, num_negatives=32000):
        self.bank = torch.FloatTensor(data_size, feature_size).zero_()
        self.bank = F.normalize(self.bank, dim=-1, p=2)
        self.num_negatives = num_negatives
        self.data_size = data_size
        self.size = data_size
        self.m = momentum
        self.ptr = 0 
        
    def update_vectors(self, indices, new_vectors):
        new_vectors = F.normalize(new_vectors, p=2, dim=-1)
        self.bank[indices] = self.m * self.bank[indices] + (1 - self.m) * new_vectors.detach().cpu()

    def get_vectors(self, exclude_idx):
        indices = torch.tensor([i for i in torch.randperm(self.data_size) if i not in exclude_idx]).long()
        return self.bank[indices[:self.num_negatives]]


class EncoderModel(nn.Module):
    
    def __init__(self, encoder, encoder_dim, projection_dim, patch_size, num_patches):
        super(EncoderModel, self).__init__()
        self.encoder = encoder 
        self.patch_size = patch_size
        self.f_proj_head = nn.Linear(encoder_dim, projection_dim)
        self.g_proj_head_initial = nn.Linear(encoder_dim, projection_dim)
        self.g_proj_head_final = nn.Linear(projection_dim * num_patches, projection_dim)

    def forward(self, imgs):
        # Patchwise feature extraction and random concatenation
        # for Jigsaw task is performed in this module
        # imgs is non-transformed image tensor of shape (bs, c, h, w)
        patch_features = []
        bs, c, h, w = imgs.size()
        w_offsets = [(i*self.patch_size, (i+1)*self.patch_size) for i in range(w // self.patch_size)]
        h_offsets = [(i*self.patch_size, (i+1)*self.patch_size) for i in range(h // self.patch_size)]
        for x1, x2 in w_offsets:
            for y1, y2 in h_offsets:
                patch_features.append(self.g_proj_head_initial(self.encoder(imgs[:, :, y1:y2, x1:x2])))               
        patch_features = torch.cat(patch_features, 1)
        patch_features = self.g_proj_head_final(patch_features)
        image_features = self.f_proj_head(self.encoder(imgs))
        return image_features, patch_features 


class PretextInvariantRepresentationModel:

    def __init__(self, args):
        assert args["arch"] in NETWORKS.keys(), f"Expected 'arch' to be one of {list(NETWORKS.keys())}"
        output_root = os.path.join("outputs/pirl", args["arch"])
        
        self.config, self.output_dir, self.logger, self.device = common.initialize_experiment(args, output_root)
        self.train_loader, self.test_loader = data_utils.get_indexed_dataloaders(**self.config["data"])
        run = wandb.init(**self.config["wandb"])
        self.logger.write("Wandb url: {}".format(run.get_url()), mode="info")

        encoder, encoder_dim = NETWORKS[args["arch"]].values()
        self.model = EncoderModel(
            encoder(**self.config["encoder"]), encoder_dim, self.config["proj_dim"], self.config["patch_size"], self.config["num_patches"]).to(self.device)
        self.memory_bank = MemoryBank(len(self.train_loader.dataset), self.config["proj_dim"], self.config["momentum"], self.config["num_negatives"])

        self.optim = train_utils.get_optimizer(self.config["optimizer"], params=self.model.parameters())
        self.scheduler, self.warmup_epochs = train_utils.get_scheduler({**self.config["scheduler"], "epochs": self.config["epochs"]}, optimizer=self.optim)
        if self.warmup_epochs > 0:
            self.warmup_rate = (self.config["optimizer"]["lr"] - 1e-12) / self.warmup_epochs

        self.loss_fn = losses.PirlLoss(**self.config["loss_fn"])
        self.best_metric = 0
        
        if args["load"] is not None:
            self.load_checkpoint(args["load"])

    def save_checkpoint(self):
        state = {"encoder": self.model.state_dict()}
        torch.save(state, os.path.join(self.output_dir, "best_model.pt"))

    def load_checkpoint(self, ckpt_dir):
        if os.path.exists(os.path.join(ckpt_dir, "encoder")):
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

    def train_step(self, batch):
        indices, img = batch["index"], batch["img"].to(self.device)
        img_features, patch_features = self.model(img)
        loss = self.loss_fn(img_features, patch_features, self.memory_bank.get_vectors(indices).to(self.device))
        self.memory_bank.update_vectors(indices, img_features)

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
                z = self.model(img)[0]
                z = F.normalize(z, dim=-1, p=2).detach().cpu().numpy()
                fvecs.append(z), gt.append(trg)
                common.progress_bar(progress=(step+1)/len(self.train_loader), desc="Building train features")
            print()

        elif split == "test":
            for step, batch in enumerate(self.test_loader):
                img, trg = batch["img"].to(self.device), batch["label"].detach().cpu().numpy()
                z = self.model(img)[0]
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