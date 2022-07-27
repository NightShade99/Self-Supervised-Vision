
import os
import yaml
import wandb
import pickle
import argparse
import functools
from typing import Any
from models import resnet
from torch.utils import data
from datetime import datetime as dt
from utils import dataset, common, optimization

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
    trainset, valset = dataset.get_datasets(**cfg['data'])
    
    if 'cifar' in cfg['data']['name']:
        inp_shape = (32, 32, 3)
        num_classes = 10
    elif 'imagenet' in cfg['data']['name']:
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
        small_images=('cifar' in cfg['data']['name']),
        use_classifier=(args.algo == 'classification')
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
    def compute_metrics(logits, labels):
        labels_onehot = jax.nn.one_hot(labels, num_classes=num_classes)
        loss = optax.softmax_cross_entropy(logits, labels_onehot)
        loss = jnp.mean(loss)

        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        metrics = {'loss': loss, 'accuracy': accuracy}
        
        metrics = jax.lax.pmean(metrics, axis_name='device')
        metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
        return metrics
    
    @functools.partial(jax.pmap, axis_name='device')
    def train_step(batch, state):
        images, labels = batch
        images, labels = images[0], labels[0]

        def loss_fn(params):
            outputs, new_state = state.apply_fn(
                {'params': params, 'batch_stats': state.batch_stats},
                images, train=True, mutable=['batch_stats']
            )
            logits = nn.log_softmax(outputs['outputs'], -1)
            labels_onehot = jax.nn.one_hot(labels, num_classes=num_classes)
            loss = optax.softmax_cross_entropy(logits, labels_onehot)
            loss = jnp.mean(loss)

            # L2 weight decay
            weight_penalty_params = jax.tree_util.tree_leaves(params)
            weight_l2 = sum([jnp.sum(x ** 2) for x in weight_penalty_params if x.ndim > 1])
            loss = loss + args.weight_decay * 0.5 * weight_l2
            return loss, (logits, new_state)
        
        aux, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name='device')
        
        logits, new_state = aux[1]
        new_state = state.apply_gradients(grads=grads, batch_stats=new_state['batch_stats'])
        metrics = compute_metrics(logits, labels)
        return new_state, metrics

    @functools.partial(jax.pmap, axis_name='device')
    def eval_step(batch, state):
        images, labels = batch
        images, labels = images[0], labels[0]

        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        outputs = state.apply_fn(variables, images, train=False, mutable=False)
        logits = nn.log_softmax(outputs['outputs'], -1)
        metrics = compute_metrics(logits, labels)
        return metrics
    
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
        for step, batch in enumerate(val_loader):
            metrics = eval_step(batch, state)
            val_meter.add(metrics)
            
            if jax.process_index() == 0:
                common.pbar(
                    progress=(step+1) / len(val_loader),
                    desc='[EVAL] Epoch {}'.format(epoch),
                    status=val_meter.msg()
                )
        
        if jax.process_index() == 0:
            logger.write("Epoch {} {}".format(epoch, val_meter.msg()), mode='val')
            avg_metrics = val_meter.avg()
            
            if args.wandb:
                wandb.log({
                    'eval/loss': avg_metrics['loss'],
                    'eval/accuracy': avg_metrics['accuracy'],
                    'epoch': epoch
                })
                
            if avg_metrics['accuracy'] > best_val_acc:
                best_val_acc = avg_metrics['accuracy']
                save_checkpoint(state, workdir)
                
    # Wait until computations are over to close
    jax.random.normal(jax.random.PRNGKey(args.seed), ()).block_until_ready()
    

  
if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_file', type=str, required=True)
    ap.add_argument('--expt_name', type=str, required=True)
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--algo', type=str, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--load', type=str, default=None)
    ap.add_argument('--wandb', action='store_true', default=False)
    ap.add_argument('--batch_size', type=int, default=100)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--weight_decay', type=float, default=1e-05)
    ap.add_argument('--train_epochs', type=int, default=100)
    args = ap.parse_args()
    
    main(args)
