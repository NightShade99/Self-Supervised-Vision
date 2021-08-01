
import os
import wandb
import numpy as np 
import tensorflow as tf
import tensorflow.keras.layers as nn 
import tensorflow.keras.utils as tf_utils

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


class ProjectionHead(tf.keras.Model):

    def __init__(self, input_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Dense(input_dim)
        self.bn1 = nn.BatchNormalization()
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(output_dim)
        self.bn2 = nn.BatchNormalization()

    def __call__(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return x


class SimCLR:

    def __init__(self, args):
        assert args["arch"] in NETWORKS.keys(), f"Expected 'arch' to be one of {list(NETWORKS.keys())}"
        output_root = os.path.join("outputs/simclr", args["arch"])
        
        self.config, self.output_dir, self.logger = common.initialize_experiment(args, output_root)
        self.train_loader, self.test_loader = data_utils.get_simclr_dataloaders(**self.config["data"])
        run = wandb.init(**self.config["wandb"])
        self.logger.write("Wandb url: {}".format(run.get_url()), mode="info")

        encoder, encoder_dim = NETWORKS[args["arch"]].values()
        self.encoder = encoder(**self.config["encoder"])
        self.proj_head = ProjectionHead(encoder_dim, self.config["projection_head"]["proj_dim"])
        self.optim = train_utils.get_optimizer(self.config["optimizer"])
        self.scheduler = train_utils.get_scheduler(self.config["scheduler"])
        self.warmup_epochs = self.config["warmup_epochs"]
        if self.warmup_epochs > 0:
            self.warmup_rate = (self.config["optimizer"]["lr"] - 1e-12) / self.warmup_epochs

        self.loss_fn = losses.SimclrLoss(**self.config["loss_fn"])
        self.best_metric = 0
        
        if args["load"] is not None:
            self.load_checkpoint(args["load"])

    def save_checkpoint(self):
        self.encoder.save_weights(os.path.join(self.output_dir, "encoder"))
        self.proj_head.save_weights(os.path.join(self.output_dir, "proj_head"))

    def load_checkpoint(self, ckpt_dir):
        if os.path.exists(os.path.join(ckpt_dir, "encoder")):
            self.encoder.load_weights(os.path.join(ckpt_dir, "encoder"))
            self.proj_head.load_weights(os.path.join(ckpt_dir, "proj_head"))
            self.logger.print(f"Successfully loaded model from {ckpt_dir}")
        else:
            raise NotImplementedError(f"Could not find saved checkpoint at {ckpt_dir}")

    def adjust_learning_rate(self, epoch):
        if epoch <= self.warmup_epochs:
            new_lr = 1e-12 + epoch * self.warmup_rate
        else:
            new_lr = self.scheduler(epoch)
        self.optim.lr.assign(new_lr)

    def train_step(self, batch):
        img_1, img_2 = batch["aug_1"], batch["aug_2"]
        with tf.GradientTape() as tape:
            z_1 = self.proj_head(self.encoder(img_1))
            z_2 = self.proj_head(self.encoder(img_2))
            loss = self.loss_fn(z_1, z_2)
        
        variables = (self.encoder.trainable_variables + self.proj_head.trainable_variables)
        grads = tape.gradient(loss, variables)
        self.optim.apply_gradients(zip(grads, variables))
        return {"loss": float(loss)}

    def knn_validate(self):
        fvecs, gt = self.build_features(split="test")
        acc = eval_utils.compute_neighbor_accuracy(fvecs, gt)
        return acc

    def build_features(self, split="train"):
        fvecs, gt = [], []

        if split == "train":
            for step in range(len(self.train_loader)):
                batch = self.train_loader.get()
                img, trg = batch["img"], batch["label"]
                z = self.proj_head(self.encoder(img))
                z = tf_utils.normalize(z, axis=-1, order=2)
                fvecs.append(z), gt.append(trg)
                common.progress_bar(progress=(step+1)/len(self.train_loader), desc="Building train features")

        elif split == "test":
            for step in range(len(self.test_loader)):
                batch = self.test_loader.get()
                img, trg = batch["img"], batch["trg"]
                z = self.proj_head(self.encoder(img))
                z = tf_utils.normalize(z, axis=-1, order=2)
                fvecs.append(z), gt.append(trg)
                common.progress_bar(progress=(step+1)/len(self.test_loader), desc="Building test features")
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
            num_classes = 10
        )
        self.logger.write("Test linear eval accuracy: {:.4f}".format(test_linear_acc), mode="info")

    def train(self):
        print("\n[INFO] Beginning training.")
        for epoch in range(1, self.config["epochs"]+1):
            train_meter = common.AverageMeter()
            desc_str = "Epoch {:3d}/{:3d}".format(epoch, self.config["epochs"])

            for step in range(len(self.train_loader)):
                train_metrics = self.train_step(batch=self.train_loader.get())
                wandb.log({"Train loss": train_metrics["loss"]})
                train_meter.add(train_metrics)
                common.progress_bar(progress=(step+1)/len(self.train_loader), desc=desc_str, status=train_meter.return_msg())

            self.logger.write("Epoch {:3d}/{:3d} ".format(epoch, self.config["epochs"]) + train_meter.return_msg(), mode="train")
            self.adjust_learning_rate(epoch)

            if epoch % self.config["eval_every"] == 0:
                knn_acc = self.knn_validate()
                self.logger.record("Epoch {:3d}/{:3d} [accuracy] {:.4f}".format(epoch, self.config["epochs"], knn_acc), mode="val")
                
                if knn_acc > self.best_metric:
                    self.best_metric = knn_acc
                    self.save_checkpoint()
        print()
        self.logger.print("Completed training. Beginning linear evaluation.", mode="info")
        self.perform_linear_eval()        
