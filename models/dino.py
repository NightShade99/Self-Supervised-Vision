
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
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.fc_out = nn.utils.weight_norm(nn.Linear(hidden_dim, projection_dim))

    def forward(self, x):
        x = self.proj_head(self.encoder(x))
        x = F.normalize(x, dim=-1, p=2)
        x = self.fc_out(x)
        return x


class DistillationWithNoLabels:

    def __init__(self, args):
        assert args["arch"] in NETWORKS.keys(), f"Expected 'arch' to be one of {list(NETWORKS.keys())}"
        output_root = os.path.join("outputs/dino", args["arch"])
            
        self.config, self.output_dir, self.logger, self.device = common.initialize_experiment(args, output_root)
        self.train_loader, self.test_loader = data_utils.get_multicrop_dataloaders(**self.config["data"])
        run = wandb.init(**self.config["wandb"])
        self.logger.write("Wandb url: {}".format(run.get_url()), mode="info")

        encoder, encoder_dim = NETWORKS[args["arch"]].values()
        self.student_model = EncoderModel(
            encoder = encoder(self.config["encoder"]), 
            encoder_dim = self.config["encoder"]["hidden_dim"], 
            hidden_dim = self.config["proj_head"]["hidden_dim"], 
            projection_dim = self.config["proj_head"]["proj_dim"]).to(self.device)
        self.teacher_model = EncoderModel(
            encoder = encoder(self.config["encoder"]), 
            encoder_dim = self.config["encoder"]["hidden_dim"], 
            hidden_dim = self.config["proj_head"]["hidden_dim"], 
            projection_dim = self.config["proj_head"]["proj_dim"]).to(self.device)

        self.teacher_center = torch.randn(1, self.config["proj_head"]["proj_dim"]).to(self.device)
        self.temp_teacher = self.config.get("teacher_temp_lower", 0.04)
        self.temp_student = self.config.get("student_temp", 0.1)
        self.m = self.config.get("center_momentum", 0.9)
        for p in self.teacher_model.parameters():
            p.requires_grad = False 
        
        # Gradient clipping if specified
        if self.config.get("gradient_clip", None) is not None:
            clip_val = self.config["gradient_clip"]
            for p in self.student_model.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -clip_val, clip_val))

        self.optim = train_utils.get_optimizer(self.config["optimizer"], self.student_model.parameters())
        self.scheduler, self.warmup_epochs = train_utils.get_scheduler({**self.config["scheduler"], "epochs": self.config["epochs"]}, self.optim)
        if self.warmup_epochs > 0:
            self.warmup_rate = (self.config["optimizer"]["lr"] - 1e-12) / self.warmup_epochs 

        self.loss_fn = losses.DinoLoss()
        self.best_metric = 0

        if args["load"] is not None:
            self.load_checkpoint(args["load"])

    def save_checkpoint(self):
        state = {"encoder": self.student_model.state_dict()}
        torch.save(state, os.path.join(self.output_dir, "best_model.pt"))

    def load_checkpoint(self, ckpt_dir):
        if os.path.exists(os.path.join(ckpt_dir, "encoder")):
            state = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=self.device)
            self.student_model.load_state_dict(state["encoder"])
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
    def update_temperature(self, epoch):
        temp_t_upper, temp_t_lower = self.config.get("teacher_temp_upper", 0.07), self.config.get("teacher_temp_lower", 0.04)
        self.temp_student = self.config.get("student_temp", 0.1)
        if epoch <= self.config.get("temp_warmup_epochs", 30):
            self.temp_teacher = temp_t_lower + (temp_t_upper - temp_t_lower) * (epoch / self.config.get("temp_warmup_epochs", 30))
        else:
            self.temp_teacher = temp_t_upper
        
    @torch.no_grad()
    def update_weight_decay(self, epoch):
        wd_upper, wd_lower = self.config.get("weight_decay_upper", 0.4), self.config.get("weight_decay_lower", 0.04)
        weight_decay = wd_upper - (wd_upper - wd_lower) * (math.cos(math.pi * epoch / self.config["epochs"]) + 1) / 2
        for group in self.optim.param_groups:
            group["weight_decay"] = weight_decay

    @torch.no_grad()
    def update_teacher_model(self, epoch):
        lbd_upper, lbd_lower = self.config.get("lambda_upper", 1.0), self.config.get("lambda_lower", 0.996)
        lbd = lbd_upper - (lbd_upper - lbd_lower) * (math.cos(math.pi * epoch / self.config["epochs"]) + 1) / 2
        for t_param, s_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
            t_param.data = lbd * t_param.data + (1.0 - lbd) * s_param.data

    @torch.no_grad()
    def update_teacher_center(self, teacher_fvecs):
        if self.teacher_center is None:
            self.teacher_center = teacher_fvecs.mean(0)
        else:
            self.teacher_center = self.m * self.teacher_center + (1 - self.m) * teacher_fvecs.mean(0)

    def train_step(self, batch):
        global_1, global_2 = batch["global_1"].to(self.device), batch["global_2"].to(self.device)
        local_1, local_2 = batch["local_1"].to(self.device), batch["local_2"].to(self.device)
        bs, vg, c, hg, wg = global_1.size()
        bs, vl, c, hl, wl = local_1.size()
        global_1, global_2 = global_1.view(-1, c, hg, wg), global_2.view(-1, c, hg, wg)
        local_1, local_2 = local_1.view(-1, c, hl, wl), local_2.view(-1, c, hl, wl)

        student_g1, student_g2 = self.student_model(global_1), self.student_model(global_2)                         # (2, K), (2, K)
        student_l1, student_l2 = self.student_model(local_1), self.student_model(local_2)                           # (V, K), (V, K)
        student_1, student_2 = torch.cat((student_g1, student_l1), 0), torch.cat((student_g2, student_l2), 0)
        with torch.no_grad():
            teacher_1, teacher_2 = self.teacher_model(global_1), self.teacher_model(global_2)                       # (2, K), (2, K)

        # Update center and reshape features back to (batch_size, num_views, feature_dim)
        student_1, student_2 = student_1.view(bs, vg+vl, -1), student_2.view(bs, vg+vl, -1)
        teacher_1, teacher_2 = teacher_1.view(bs, vg, -1), teacher_2.view(bs, vg, -1)

        loss_1 = 0.5 * self.loss_fn(teacher_1, student_2, self.temp_student, self.temp_teacher, self.teacher_center)
        loss_2 = 0.5 * self.loss_fn(teacher_2, student_1, self.temp_student, self.temp_teacher, self.teacher_center)
        loss = loss_1 + loss_2
        self.update_teacher_center(torch.cat((teacher_1.view(bs*vg, -1), teacher_2.view(bs*vg, -1)), 0))

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
                z = self.student_model(img)
                fvecs.append(z), gt.append(trg)
                common.progress_bar(progress=(step+1)/len(self.train_loader), desc="Building train features")
            print()

        elif split == "test":
            for step, batch in enumerate(self.test_loader):
                img, trg = batch["img"].to(self.device), batch["label"].detach().cpu().numpy()
                z = self.student_model(img)
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
            self.update_teacher_model(epoch)
            self.update_weight_decay(epoch)
            self.update_temperature(epoch)
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
        