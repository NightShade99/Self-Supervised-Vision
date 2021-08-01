
import faiss 
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
from . import data_utils, losses


def compute_neighbor_accuracy(fvecs, labels, k=20):
    index = faiss.IndexFlatIP(fvecs.shape[1])
    index.add(fvecs.astype(np.float32))
    _, neighbor_idx = index.search(fvecs, top_k=k+1)

    anchor_targets = np.repeat(targets.reshape(-1, 1), k, axis=1)
    neighbor_targets = np.take(targets, neighbor_idx[:, 1:], axis=0)
    accuracy = np.mean(anchor_targets == neighbor_targets)
    return accuracy


def linear_evaluation(config, train_data, test_data, num_classes, device):
    train_loader = data_utils.get_feature_dataloaders(train_data["fvecs"], train_data["labels"], batch_size=config["batch_size"])
    test_loader = data_utils.get_feature_dataloaders(test_data["fvecs"], test_data["labels"], batch_size=config["batch_size"])

    clf_head = nn.Linear(config["input_dim"], num_classes).to(device)
    clf_optim = optim.SGD(params=clf_head.parameters(), lr=config["lr"], momentum=config.get("momentum", 0.9), weight_decay=config.get("weight_decay", 1e-06))
    clf_sched = lr_sched.CosineAnnealingLR(clf_optim, T_max=config["epochs"], eta_min=0.0, last_epoch=-1)
    loss_fn = nn.NLLLoss()
    
    for epoch in range(1, config["linear_eval_epochs"]+1):
        train_meter = common.AverageMeter()
        test_meter = common.AverageMeter()
        desc_str = "Epoch {:2d}/{:2d}".format(epoch, config["epochs"])

        for step, batch in enumerate(train_loader):
            fvecs, labels = batch["features"].to(device), batch["label"].to(device)
            logits = clf_head(features)
            loss = loss_fn(F.log_softmax(logits, dim=-1), labels)
            acc = torch.mean(F.softmax(logits, dim=-1).argmax(dim=-1) == labels)
            
            clf_optim.zero_grad()
            loss.backward()
            clf_optim.step()
            train_meter.add({"loss": float(loss), "accuracy": float(acc)})
            common.progress_bar(progress=(step+1)/len(train_loader), desc=desc_str, status=train_meter.return_msg())

        for step, batch in enumerate(test_loader):
            fvecs, labels = batch["features"].to(device), batch["label"].to(device)
            with torch.no_grad():
                logits = clf_head(features)
            loss = loss_fn(F.log_softmax(logits, dim=-1), labels)
            acc = torch.mean(F.softmax(logits, dim=-1).argmax(dim=-1) == labels)
            test_meter.add({"loss": float(loss), "accuracy": float(acc)})
            common.progress_bar(progress=(step+1)/len(test_loader), desc=desc_str, status=test_meter.return_msg())

        # Reduce learning rate
        clf_sched.step()

    print("\nCompleted linear evaluation. Average validation accuracy is {:.2f}%".format(100 * test_meter.return_metrics()["accuracy"]))
    return test_meter.return_metrics()["accuracy"]