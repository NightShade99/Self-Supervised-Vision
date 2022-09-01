
import os
import yaml
import wandb
import argparse
import functools
import numpy as np
from typing import Any
from models import resnet
from torch.utils import data
from datetime import datetime as dt
from utils import dataset, common, evaluation, optimization

import jax 
import optax
import jax.numpy as jnp
import flax.linen as nn
from flax import jax_utils
from flax.training import train_state
from flax.training import checkpoints

os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

MODELS = {
    'resnet18': resnet.ResNet18,
    'resnet34': resnet.ResNet34,
    'resnet50': resnet.ResNet50,
    'resnet101': resnet.ResNet101,
    'resnet152': resnet.ResNet152,
    'resnet200': resnet.ResNet200
}

class TrainState(train_state.TrainState):
    batch_stats: Any = None
    

def main(args):
    # Device information
    print("\nTraining platform: {}".format(jax.devices()[0].platform))
    print("Number of available devices: {}".format(jax.device_count()))
    
    # Load config file
    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Random key
    rng = jax.random.PRNGKey(args.seed)
    param_rng, _ = jax.random.split(rng)
    init_rng = {'params': param_rng}
        
    # Datasets and dataloaders
    trainset = dataset.DoubleAugmentDataset(**cfg['data']['train'])
    valset = dataset.DoubleAugmentDataset(**cfg['data']['val'])
    
    if 'cifar' in cfg['data']['train']['base_dataset']:
        inp_shape = (32, 32, 3)
        num_classes = 10
    elif 'imagenet' in cfg['data']['train']['base_dataset']:
        inp_shape = (224, 224, 3)
        num_classes = 1000
    
    train_loader = data.DataLoader(
        trainset, 
        batch_size=args.batch_size * jax.device_count(),
        num_workers=args.num_workers,
        collate_fn=dataset.numpy_collate, 
        shuffle=True,
        drop_last=True,
        persistent_workers=True
    )
    val_loader = data.DataLoader(
        valset,
        batch_size=args.batch_size * jax.device_count(),
        num_workers=args.num_workers,
        collate_fn=dataset.numpy_collate,
        shuffle=False,
        drop_last=False,
        persistent_workers=True
    )
    
    # Model initialization
    assert args.model in MODELS, f'Invalid model "{args.model}"'
    
    model = MODELS[args.model](
        num_classes=num_classes,
        small_images=('cifar' in cfg['data']['train']['base_dataset']),
        use_classifier=(args.algo == 'supervised'),
        projection_dim=cfg['model']['projection_dim']
    )
    variables = model.init(init_rng, jnp.ones((1, *inp_shape), dtype=jnp.float32))
    
    # Optimizer, learning rate schedule and train state
    lr_schedule = optimization.build_lr_schedule(**cfg['lr_schedule'])
    optimizer = optimization.build_optimizer(cfg['optimizer']['name'], lr_schedule=lr_schedule)
    
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        batch_stats=variables['batch_stats'],
        tx=optimizer
    )
    
    # Logging and checkpoint loading
    if args.load is None:
        expt_time = dt.now().strftime('%d-%m-%Y_%H-%M')
        workdir = os.path.join('outputs', args.expt_name, args.model, expt_time)
        os.makedirs(workdir, exist_ok=True)
        
        logger = common.Logger(workdir)
        if args.wandb:
            run = wandb.init(project='jax-classification', name=expt_time)
            logger.write("Wandb URL: {}".format(run.get_url()))
    else:
        state = checkpoints.restore(args.load, state)
            
    state = jax_utils.replicate(state)
    best_val_acc = -float("inf")
    
    # Functions for training and evaluation
    # These will be pmapped later for computation across devices
    @functools.partial(jax.pmap, axis_name='device')
    def train_step(batch, state):
        aug_1, aug_2, _, _ = batch
        aug_1, aug_2 = aug_1[0], aug_2[0]

        def loss_fn(params):
            outputs_1, new_state = state.apply_fn(
                {'params': params, 'batch_stats': state.batch_stats},
                aug_1, train=True, mutable=['batch_stats']
            )
            outputs_2, new_state = state.apply_fn(
                {'params': params, 'batch_stats': state.batch_stats},
                aug_2, train=True, mutable=['batch_stats']
            )
            logits_1, logits_2 = outputs_1['outputs'], outputs_2['outputs']
            
            # Normalize logits
            logits_1 = logits_1 / jnp.linalg.norm(logits_1, ord=2, axis=-1)
            logits_2 = logits_2 / jnp.linalg.norm(logits_2, ord=2, axis=-1)
            
            bs = logits_1.shape[0]
            labels = jnp.zeros((2 * bs,), dtype=jnp.int32)
            labels = jax.nn.one_hot(labels, num_classes=2)

            mask = jnp.ones((bs, bs), dtype=bool)
            mask.at[jnp.diag_indices_from(mask)].set(0)
            
            logits_11 = jnp.matmul(logits_1, logits_1.transpose()) / cfg['model']['temperature']
            logits_12 = jnp.matmul(logits_1, logits_2.transpose()) / cfg['model']['temperature']
            logits_21 = jnp.matmul(logits_2, logits_1.transpose()) / cfg['model']['temperature']
            logits_22 = jnp.matmul(logits_2, logits_2.transpose()) / cfg['model']['temperature']
                
            logits_12_pos = logits_12[jnp.logical_not(mask)]
            logits_21_pos = logits_21[jnp.logical_not(mask)]
            logits_11_neg = logits_11[mask].reshape(bs, -1)
            logits_12_neg = logits_12[mask].reshape(bs, -1)
            logits_21_neg = logits_21[mask].reshape(bs, -1)
            logits_22_neg = logits_22[mask].reshape(bs, -1)
            
            pos = jnp.concatenate((logits_12_pos, logits_21_pos), axis=0)[:, None]
            neg_1 = jnp.concatenate((logits_11_neg, logits_12_neg), axis=1)
            neg_2 = jnp.concatenate((logits_21_neg, logits_22_neg), axis=1)
            neg = jnp.concatenate((neg_1, neg_2), axis=0)
            
            logits = jnp.concatenate((pos, neg), axis=1)
            logprobs = nn.log_softmax(logits, axis=1)
            loss = optax.softmax_cross_entropy(logprobs, labels)
            loss = jnp.mean(loss)
            
            # L2 weight decay
            weight_penalty_params = jax.tree_util.tree_leaves(params)
            weight_l2 = sum([jnp.sum(x ** 2) for x in weight_penalty_params if x.ndim > 1])
            loss = loss + args.weight_decay * 0.5 * weight_l2
            return loss, (logits, new_state)
        
        aux, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name='device')
        
        _, new_state = aux[1]
        new_state = state.apply_gradients(grads=grads, batch_stats=new_state['batch_stats'])
        return new_state, metrics
    
    def sync_batch_stats(state):
        cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
        return state.replace(batch_stats=cross_replica_mean(state.batch_stats))
    
    def save_checkpoint(state, workdir):
        if jax.process_index() == 0:
            state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
            step = int(state.step)
            checkpoints.save_checkpoint(workdir, state, step, keep=3)
    
    # Training loop
    for epoch in range(1, args.train_epochs+1):
        train_meter = common.AverageMeter()
        val_meter = common.AverageMeter()
        
        for step, batch in enumerate(train_loader):
            state, metrics = train_step(batch, state)            
            train_meter.add(metrics)
            
            if jax.process_index() == 0:
                common.pbar(
                    progress=(step+1) / len(train_loader),
                    desc='[TRAIN] Epoch {}'.format(epoch),
                    status=train_meter.msg()
                )
                if args.wandb:
                    wandb.log({'train/loss': metrics['loss']})
                
        state = sync_batch_stats(state)
        
        if jax.process_index() == 0:
            logger.write("Epoch {} {}".format(epoch, train_meter.msg()), mode='train')
            if args.wandb:
                wandb.log({'train/accuracy': train_meter.avg()['loss'], 'epoch': epoch})
                
        # Evaluation
        features, targets = [], []
        
        for step, batch in enumerate(val_loader):
            _, _, images, labels = batch 
            images, labels = images[0], labels[0]
            
            outputs, _ = state.apply_fn(
                {'params': state.params, 'batch_stats': state.batch_stats}, 
                images, train=False
            )
            fvecs = outputs['outputs']
            fvecs = fvecs / jnp.linalg.norm(fvecs, ord=2, axis=-1)
            
            features.append(np.asarray(fvecs))
            targets.append(np.asarray(labels))
            
            if jax.process_index() == 0:
                common.pbar(
                    progress=(step+1) / len(val_loader),
                    desc='[EVAL] Epoch {}'.format(epoch),
                    status=val_meter.msg()
                )
                
        fvecs = np.concatenate(features).astype(np.float32)
        targets = np.concatenate(targets).astype(np.int32)
        knn_acc = evaluation.compute_neighbor_accuracy(fvecs, targets)
        
        if jax.process_index() == 0:
            logger.write("Epoch {} [knn accuracy] {}".format(epoch, knn_acc), mode='val')
            
            if args.wandb:
                wandb.log({
                    'eval/knn_accuracy': knn_acc,
                    'epoch': epoch
                })
                
            if knn_acc > best_val_acc:
                best_val_acc = knn_acc
                save_checkpoint(state, workdir)
                
    # Wait until computations are over to close
    jax.random.normal(jax.random.PRNGKey(args.seed), ()).block_until_ready()