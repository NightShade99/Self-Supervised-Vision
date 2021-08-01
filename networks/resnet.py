
import numpy as np 
import tensorflow as tf 
import tensorflow.keras.layers as nn 
import tensorflow_addons.layers as tfa_nn
import tensorflow.keras.initializers as init

INIT_HE_NORMAL = init.HeNormal()


def conv3x3(out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2D(
        filters = out_planes, 
        kernel_size = (3, 3), 
        strides = (stride, stride), 
        dilation_rate = (dilation, dilation),
        padding = "same", 
        groups = groups, 
        use_bias = False,
        kernel_initializer = INIT_HE_NORMAL
    )

def conv1x1(out_planes, stride=1):
    return nn.Conv2D(
        filters = out_planes,
        kernel_size = (1, 1),
        strides = (stride, stride), 
        use_bias = False,
        kernel_initializer = INIT_HE_NORMAL
    )


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        assert groups == 1 and base_width == 64, "BasicBlock only supports groups=1 and base_width=64"
        assert 0 <= dilation <= 1, "Dilation > 1 not supported for BasicBlock" 

        self.conv1 = conv3x3(planes, stride)
        self.bn1 = nn.BatchNormalization()
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes)
        self.bn2 = nn.BatchNormalization()
        self.downsample = downsample 
        self.stride = stride 

    def __call__(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity 
        out = self.relu(out)
        return out 

        
class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(Bottleneck, self).__init__()
        width = int(planes * base_width / 64.) * groups

        self.conv1 = conv1x1(width)
        self.bn1 = nn.BatchNormalization() 
        self.conv2 = conv3x3(width, stride, groups, dilation)
        self.bn2 = nn.BatchNormalization()
        self.conv3 = conv1x1(planes * self.expansion)
        self.bn3 = nn.BatchNormalization() 
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride 

    def __call__(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(tf.keras.Model):

    def __init__(self, block, layers, groups=1, width_per_group=64, replace_stride_with_dilation=None, reduce_bottom=False):
        super(ResNet, self).__init__()
        self.base_width = width_per_group 
        self.groups = groups 
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if not reduce_bottom:
            self.conv1 = nn.Conv2D(self.inplanes, kernel_size=(7, 7), strides=(2, 2), padding="same", use_bias=False, kernel_initializer=INIT_HE_NORMAL)
        else:
            self.conv1 = nn.Conv2D(self.inplanes, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False, kernel_initializer=INIT_HE_NORMAL)
        self.bn1 = nn.BatchNormalization()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]) 
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]) 
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = tfa_nn.AdaptiveAveragePooling2D(output_size=(1, 1), data_format='channels_last')
        self.flatten = nn.Flatten()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None 
        previous_dilation = self.dilation 
        if dilate:
            self.dilation *= stride 
            stride = 1
        if stride != 1 or self.inplanes * planes * block.expansion:
            downsample = tf.keras.models.Sequential([
                conv1x1(planes * block.expansion, stride),
                nn.BatchNormalization()
            ])

        layers = []
        layers.append(block(planes, stride, downsample, self.groups, self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion 
        for _ in range(1, blocks):
            layers.append(block(planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation))
        return tf.keras.models.Sequential(layers)

    def __call__(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model 

def resnet18(**kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet50(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnext50_32x4d(**kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnext101_32x8d(**kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)

def wide_resnet50_2(**kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)

def wide_resnet101_2(**kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


if __name__ == "__main__":

    model1 = resnet18(reduce_bottom=True, num_classes=10)
    model2 = resnext50_32x4d(reduce_bottom=True, num_classes=10)
    model3 = wide_resnet50_2(reduce_bottom=True, num_classes=10)
    a = np.random.uniform(0, 1, size=(8, 32, 32, 3))
    out1, out2, out3 = model1(a), model2(a), model3(a)
    print(out1.shape, out2.shape, out3.shape)