
import optax


def build_optimizer(name, lr_schedule):
    if name == 'sgd':
        return optax.sgd(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
    elif name == 'rmsprop':
        return optax.rmsprop(learning_rate=lr_schedule, decay=0.9, momentum=0.9, nesterov=True)
    elif name == 'adam':
        return optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.999)
    elif name == 'lars':
        return optax.lars(learning_rate=lr_schedule, weight_decay=1e-06, momentum=0.9, nesterov=True)
    
    
def build_lr_schedule(name, base_lr, schedule_steps=0, warmup_steps=0, decay_rate=0.99):
    if name == 'cosine':
        return optax.cosine_decay_schedule(
            init_value=base_lr, decay_steps=schedule_steps, alpha=1e-10
        )
    elif name == 'exponential':
        return optax.exponential_decay(
            init_value=base_lr, transition_steps=schedule_steps, decay_rate=decay_rate
        )
    elif name == 'warmup_cosine':
        return optax.warmup_cosine_decay_schedule(
            init_value=1e-12, peak_value=base_lr, warmup_steps=warmup_steps, 
            decay_steps=schedule_steps, end_value=1e-10
        )
    elif name == 'constant':
        return optax.constant_schedule(
            value=base_lr
        )