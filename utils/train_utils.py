
import tensorflow.keras.optimizers as optim 
import tensorflow_addons.optimizers as tfa_optim
import tensorflow.keras.optimizers.schedules as lr_sched


def get_optimizer(config):
    name = config.get("name", "sgd")

    if name == "sgd":
        return tfa_optim.SGDW(
            learning_rate=config["lr"], momentum=config.get("momentum", 0.9), nesterov=config.get("nesterov", True),
            weight_decay=config.get("weight_decay", 1e-06)
        )
    elif name == "adam":
        return tfa_optim.AdamW(
            learning_rate=config["lr"], beta_1=config.get("beta_1", 0.9), beta_2=config.get("beta_2", 0.999),
            amsgrad=config.get("amsgrad", False), epsilon=config.get("epsilon", 1e-07), weight_decay=config.get("weight_decay", 1e-06)
        )
    elif name == "rmsprop":
        return optim.RMSProp(
            learning_rate=config["lr"], rho=config.get("rho", 0.9), momentum=config.get("momentum", 0.9),
            centered=config.get("centered", False), epsilon=config.get("epsilon", 1e-07)
        )
    else:
        raise ValueError(f"Unrecognized optimizer {name}, expected one of [sgd, adam, rmsprop]")


def get_scheduler(config):
    name = config.get("name", None)

    if name == "inverse_time":
        return lr_sched.InverseTimeDecay(
            initial_learning_rate=config["lr"], decay_steps=config["decay_steps"], decay_rate=config["decay_rate"]
        )
    elif name == "cosine":
        return lr_sched.CosineDecay(
            initial_learning_rate=config["lr"], decay_steps=config["decay_steps"], alpha=config.get("alpha", 0.0)
        )
    elif name == "multistep":
        return lr_sched.PiecewiseConstantDecay(
            boundaries=config["steps"], values=config["lr_vals"]
        )
    else:
        return None

    