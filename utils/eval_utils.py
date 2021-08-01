
import faiss 
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as nn 
import tensorflow.keras.losses as tf_losses
import tensorflow.keras.optimizers as optim
import tensorflow.keras.optimizers.schedules as lr_sched
from . import data_utils, losses


def compute_neighbor_accuracy(fvecs, labels, k=20):
    index = faiss.IndexFlatIP(fvecs.shape[1])
    index.add(fvecs.astype(np.float32))
    _, neighbor_idx = index.search(fvecs, top_k=k+1)

    anchor_targets = np.repeat(targets.reshape(-1, 1), k, axis=1)
    neighbor_targets = np.take(targets, neighbor_idx[:, 1:], axis=0)
    accuracy = np.mean(anchor_targets == neighbor_targets)
    return accuracy


def linear_evaluation(config, train_data, test_data, num_classes):
    train_loader = data_utils.FeatureDataLoader(train_data["fvecs"], train_data["labels"], batch_size=config["batch_size"], shuffle=True)
    test_loader = data_utils.FeatureDataLoader(test_data["fvecs"], test_data["labels"], batch_size=config["batch_size"], shuffle=False)

    clf_head = nn.Dense(num_classes)
    clf_optim = optim.SGD(learning_rate=config["learning_rate"], momentum=config.get("momentum", 0.9), nesterov=config.get("nesterov", True))
    clf_sched = lr_sched.CosineDecay(initial_learning_rate=config["learning_rate"], decay_steps=config["epochs"], alpha=0.0)
    loss_fn = tf_losses.CategoricalCrossentropy(from_logits=True)
    softmax = nn.Softmax(axis=-1)
    
    for epoch in range(1, config["linear_eval_epochs"]+1):
        train_meter = common.AverageMeter()
        test_meter = common.AverageMeter()
        desc_str = "Epoch {:2d}/{:2d}".format(epoch, config["epochs"])

        for step in range(len(train_loader)):
            with tf.GradientTape() as tape:
                fvecs, labels = train_loader.get()
                logits = clf_head(features)
                loss = loss_fn(tf.one_hot(labels, depth=num_classes), logits)
                acc = np.mean(np.argmax(softmax(logits), axis=-1) == labels)
                grads = tape.gradient(loss, clf_head.trainable_variables)
                clf_optim.apply_gradients(zip(grads, clf_head.trainable_variables))
            train_meter.add({"loss": float(loss), "accuracy": float(acc)})
            common.progress_bar(progress=(step+1)/len(train_loader), desc=desc_str, status=train_meter.return_msg())

        for step in range(len(test_loader)):
            fvecs, labels = test_loader.get()
            logits = clf_head(features)
            loss = loss_fn(tf.one_hot(labels, depth=num_classes), logits)
            acc = np.mean(np.argmax(softmax(logits), axis=-1) == labels)
            test_meter.add({"loss": float(loss), "accuracy": float(acc)})
            common.progress_bar(progress=(step+1)/len(test_loader), desc=desc_str, status=test_meter.return_msg())

        # Reduce learning rate
        clf_optim.lr.assign(clf_sched(epoch))

    print("\nCompleted linear evaluation. Average validation accuracy is {:.2f}%".format(100 * test_meter.return_metrics()["accuracy"]))
    return test_meter.return_metrics()["accuracy"]