
import jax.numpy as jnp
import flax.linen as nn

from functools import partial
from typing import Any, Callable, Sequence, Tuple

ModuleDef = Any 


class BasicBlock(nn.Module):
    filters: int 
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)
    expansion: int = 1
    
    @nn.compact
    def __call__(self, x):
        residual = x
        out = self.conv(self.filters, (3, 3), self.strides)(x)
        out = self.norm()(out)
        out = self.act(out)
        out = self.conv(self.filters, (3, 3))(out)
        out = self.norm(scale_init=nn.initializers.zeros)(out)
        
        if residual.shape != out.shape:
            residual = self.conv(self.filters * self.expansion, (1, 1), 
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)
            
        return self.act(residual + out)
    
    
class BottleneckBlock(nn.Module):
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)
    expansion: int = 4
    
    @nn.compact
    def __call__(self, x):
        residual = x
        out = self.conv(self.filters, (1, 1))(x)
        out = self.norm()(out)
        out = self.act(out)
        out = self.conv(self.filters, (3, 3), self.strides)(out)
        out = self.norm()(out)
        out = self.act(out)
        out = self.conv(self.filters * self.expansion, (1, 1))(out)
        out = self.norm(scale_init=nn.initializers.zeros)(out)
        
        if residual.shape != out.shape:
            residual = self.conv(self.filters * self.expansion, (1, 1),
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)
            
        return self.act(residual + out)
    
    
class ResNet(nn.Module):
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_classes: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv
    small_images: bool = True
    use_classifier: bool = True
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(
            self.conv, 
            use_bias=False, 
            dtype=self.dtype
        )
        norm = partial(
            nn.BatchNorm,
            use_running_average=(not train),
            momentum=0.9,
            epsilon=1e-05,
            dtype=self.dtype 
        )
        
        # Reduce base convolution for small images, like in CIFAR-10
        if self.small_images:
            x = conv(self.num_filters, (3, 3), (1, 1), padding=[(1, 1), (1, 1)], name='conv_init')(x)
        else:
            x = conv(self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name='conv_init')(x)
        
        x = norm(name='bn_init')(x)
        x = nn.relu(x)        
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
            
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2 ** i,
                    strides=strides,
                    conv=conv,
                    norm=norm,
                    act=self.act
                )(x)

        x = jnp.mean(x, axis=(1, 2))
        if self.use_classifier:
            x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        
        return {'output': x}
    
    
# Model templates
ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=BasicBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BasicBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckBlock)
ResNet101 = partial(ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckBlock)
ResNet152 = partial(ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckBlock)
ResNet200 = partial(ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckBlock)