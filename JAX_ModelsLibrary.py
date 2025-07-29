import jax.numpy as jnp
from flax import linen as nn
import json
import sys
from typing import Any

class DepthwiseConv06(nn.Module):
    @nn.compact
    def __call__(self, x, config=None, rng=None, deterministic: bool = False):
        self.sow("summary", "input", x)
        # Stem
        x1 = nn.Conv(48, (3, 3), padding='SAME')(x)
        self.sow("summary", "conv1", x1)
        x1 = nn.BatchNorm(use_running_average=deterministic)(x1)
        self.sow("summary", "bn1", x1)
        x1 = nn.silu(x1)
        self.sow("summary", "act1", x1)

        # Separable conv 1
        x2 = nn.Conv(48, (3, 3), padding='SAME', feature_group_count=48)(x1)
        self.sow("summary", "sepconv2_dw", x2)
        x2 = nn.Conv(96, (1, 1), padding='SAME')(x2)
        self.sow("summary", "sepconv2_pw", x2)
        x2 = nn.BatchNorm(use_running_average=deterministic)(x2)
        self.sow("summary", "bn2", x2)
        x2 = nn.silu(x2)
        self.sow("summary", "act2", x2)

        # Separable conv 2
        x3 = nn.Conv(96, (3, 3), padding='SAME', feature_group_count=96)(x2)
        self.sow("summary", "sepconv3_dw", x3)
        x3 = nn.Conv(96, (1, 1), padding='SAME')(x3)
        self.sow("summary", "sepconv3_pw", x3)
        x3 = nn.BatchNorm(use_running_average=deterministic)(x3)
        self.sow("summary", "bn3", x3)

        # Résidu 1
        r1 = nn.Conv(96, (1, 1), padding='SAME')(x1)
        self.sow("summary", "res_proj1", r1)
        r1 = nn.BatchNorm(use_running_average=deterministic)(r1)
        self.sow("summary", "res_bn1", r1)
        x3 = nn.silu(x3 + r1)
        self.sow("summary", "act3", x3)

        # Pool
        x3 = nn.max_pool(x3, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        self.sow("summary", "pool1", x3)

        # Separable conv 3
        x4 = nn.Conv(96, (3, 3), padding='SAME', feature_group_count=96)(x3)
        self.sow("summary", "sepconv4_dw", x4)
        x4 = nn.Conv(160, (1, 1), padding='SAME')(x4)
        self.sow("summary", "sepconv4_pw", x4)
        x4 = nn.BatchNorm(use_running_average=deterministic)(x4)
        self.sow("summary", "bn4", x4)
        x4 = nn.silu(x4)
        self.sow("summary", "act4", x4)

        # Separable conv 4
        x5 = nn.Conv(160, (3, 3), padding='SAME', feature_group_count=160)(x4)
        self.sow("summary", "sepconv5_dw", x5)
        x5 = nn.Conv(160, (1, 1), padding='SAME')(x5)
        self.sow("summary", "sepconv5_pw", x5)
        x5 = nn.BatchNorm(use_running_average=deterministic)(x5)
        self.sow("summary", "bn5", x5)

        # Résidu 2
        r2 = nn.Conv(160, (1, 1), padding='SAME')(x4)
        self.sow("summary", "res_proj2", r2)
        r2 = nn.BatchNorm(use_running_average=deterministic)(r2)
        self.sow("summary", "res_bn2", r2)
        x5 = nn.silu(x5 + r2)
        self.sow("summary", "act5", x5)

        # SE Block
        se = jnp.mean(x5, axis=(1, 2), keepdims=False)
        self.sow("summary", "se_gap", se)
        se = nn.Dense(20)(se)
        self.sow("summary", "se_dense1", se)
        se = nn.silu(se)
        self.sow("summary", "se_act1", se)
        se = nn.Dense(160)(se)
        self.sow("summary", "se_dense2", se)
        se = nn.sigmoid(se)
        self.sow("summary", "se_sigmoid", se)
        se = se[:, None, None, :]  # broadcast
        self.sow("summary", "se_broadcast", se)
        x5 = x5 * se
        self.sow("summary", "se_scaled", x5)

        # Pool 2
        x5 = nn.max_pool(x5, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        self.sow("summary", "pool2", x5)

        # Conv + sep conv
        x6 = nn.Conv(320, (1, 1), padding='SAME')(x5)
        self.sow("summary", "conv6", x6)
        x6 = nn.BatchNorm(use_running_average=deterministic)(x6)
        self.sow("summary", "bn6", x6)
        x6 = nn.silu(x6)
        self.sow("summary", "act6", x6)

        x6 = nn.Conv(320, (3, 3), padding='SAME', feature_group_count=320)(x6)
        self.sow("summary", "sepconv6_dw", x6)
        x6 = nn.Conv(320, (1, 1), padding='SAME')(x6)
        self.sow("summary", "sepconv6_pw", x6)
        x6 = nn.BatchNorm(use_running_average=deterministic)(x6)
        self.sow("summary", "bn7", x6)
        x6 = nn.silu(x6)
        self.sow("summary", "act7", x6)

        # Global avg pool
        gap = jnp.mean(x6, axis=(1, 2))
        self.sow("summary", "gap", gap)

        # FC head
        x = nn.Dense(128)(gap)
        self.sow("summary", "fc1", x)
        x = nn.silu(x)
        self.sow("summary", "fc1_relu", x)
        x = nn.Dropout(rate=0.3)(x, deterministic=deterministic, rng=rng)
        self.sow("summary", "drop1", x)
        x = nn.Dense(10)(x)
        self.sow("summary", "output", x)

        return x


def channel_shuffle(x, groups):
    """Channel shuffle operation."""
    batchsize, height, width, num_channels = x.shape
    channels_per_group = num_channels // groups
    x = x.reshape(batchsize, height, width, groups, channels_per_group)
    x = jnp.transpose(x, (0, 1, 2, 4, 3))
    x = x.reshape(batchsize, height, width, num_channels)
    return x

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    features: int
    reduction_ratio: int = 16

    @nn.compact
    def __call__(self, x):
        batch, h, w, chan = x.shape
        y = nn.avg_pool(x, window_shape=(h, w), strides=(1, 1))
        y = y.reshape((batch, chan))
        y = nn.Dense(features=self.features // self.reduction_ratio)(y)
        y = nn.silu(y)  # Swish Activation
        y = nn.Dense(features=self.features)(y)
        y = nn.sigmoid(y)
        y = y.reshape((batch, 1, 1, chan))
        return x * y

class ShuffleNetV2Block(nn.Module):
    """ShuffleNetV2 Unit with SE Block and alternative activation."""
    features: int
    strides: tuple = (1, 1)
    use_se: bool = True

    @nn.compact
    def __call__(self, x, train=False):
        in_channels = x.shape[-1]
        branch_features = in_channels // 2

        if self.strides == (1, 1):
            x1, x2 = x[:, :, :, :branch_features], x[:, :, :, branch_features:]
        else:
            x1, x2 = x, x

        x1 = nn.Conv(features=branch_features, kernel_size=(1, 1), use_bias=False)(x1)
        x1 = nn.BatchNorm(use_running_average=not train)(x1)
        x1 = nn.silu(x1)
        if self.strides == (2, 2):
            x1 = nn.avg_pool(x1, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        x2 = nn.Conv(features=branch_features, kernel_size=(1, 1), use_bias=False)(x2)
        x2 = nn.BatchNorm(use_running_average=not train)(x2)
        x2 = nn.silu(x2)
        x2 = nn.Conv(features=branch_features, kernel_size=(3, 3), strides=self.strides, padding='SAME', use_bias=False)(x2)
        x2 = nn.BatchNorm(use_running_average=not train)(x2)
        x2 = nn.silu(x2)
        x2 = nn.Conv(features=branch_features, kernel_size=(1, 1), use_bias=False)(x2)
        x2 = nn.BatchNorm(use_running_average=not train)(x2)
        x2 = nn.silu(x2)
        if self.use_se:
            x2 = SEBlock(features=branch_features)(x2)

        x = jnp.concatenate([x1, x2], axis=-1)
        x = channel_shuffle(x, groups=2)
        return x

from typing import Sequence, Tuple
class ResNetBlock(nn.Module):
    """Basic ResNet block."""
    features: int
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x, train=False):
        residual = x

        # First convolution
        x = nn.Conv(features=self.features, kernel_size=(3, 3), strides=self.strides, padding='SAME', use_bias=False, kernel_init=nn.initializers.he_normal())(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # Second convolution
        x = nn.Conv(features=self.features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False, kernel_init=nn.initializers.he_normal())(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        # Shortcut connection if strides or features change
        if residual.shape != x.shape:
            residual = nn.Conv(features=self.features, kernel_size=(1, 1), strides=self.strides, use_bias=False, kernel_init=nn.initializers.he_normal())(residual)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        # Add residual and apply ReLU
        x = x + residual
        x = nn.relu(x)
        return x

class ResNet18(nn.Module):
    dropout_rate: float = 0.5  # Dropout à 0.5 par défaut
    use_se: bool = True
    num_classes: int = 47

    @nn.compact
    def __call__(self, x, train=False):
        # Initial convolution
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME', use_bias=False, kernel_init=nn.initializers.he_normal())(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        
        # ResNet blocks
        x = ResNetBlock(features=128)(x, train)
        x = ResNetBlock(features=256)(x, train)
        #x = ResNetBlock(features=512)(x, train)

        # Global average pooling
        x = nn.avg_pool(x, window_shape=(x.shape[1], x.shape[2]), strides=(1, 1))
        x = x.reshape(x.shape[0], -1)

        # Fully connected layer
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(features=self.num_classes, kernel_init=nn.initializers.he_normal())(x)

        return x


from typing import Any

class ViT(nn.Module):
    num_classes: int
    num_layers: int = 6  # Réduction du nombre de couches
    num_heads: int = 4  # Réduction du nombre de têtes d'attention
    # hidden_dim : Il s'agit de la dimension des embeddings utilisés dans le modèle. C'est la taille des vecteurs de sortie de la couche dense après l'attention multi-tête.
    hidden_dim: int = 64  # Réduction de la dimension des embeddings
    # mlp_dim : Il s'agit de la dimension interne du réseau de neurones dense (MLP) appliqué après l'attention. Cette dimension est généralement plus grande que hidden_dim pour permettre au modèle d'apprendre des transformations plus complexes.
    mlp_dim: int = 64  # Réduction de la dimension MLP
    dropout_rate: float = 0.1
    patch_size: int = 2
    image_size: int = 32

    @nn.compact
    def __call__(self, x, train=False):
        # Patch embedding
        patch_embed = nn.Conv(features=self.hidden_dim, kernel_size=(self.patch_size, self.patch_size), strides=(self.patch_size, self.patch_size))(x)
        batch_size, height, width, channels = patch_embed.shape
        patch_embed = patch_embed.reshape(batch_size, height * width, channels)

        # Class token
        cls_token = self.param('cls_token', nn.initializers.zeros, (1, 1, self.hidden_dim))
        cls_token = jnp.tile(cls_token, [batch_size, 1, 1])
        x = jnp.concatenate([cls_token, patch_embed], axis=1)

        # Position embedding
        pos_embed = self.param('pos_embed', nn.initializers.zeros, (1, height * width + 1, self.hidden_dim))
        x = x + pos_embed

        # Transformer layers
        for _ in range(self.num_layers):
            x = nn.LayerNorm()(x)
            x = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, dropout_rate=self.dropout_rate, deterministic=not train)(x, x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
            x = x + nn.Dense(features=self.hidden_dim)(x)
            x = nn.LayerNorm()(x)
            mlp_output = nn.Dense(features=self.mlp_dim)(x)
            mlp_output = nn.gelu(mlp_output)
            mlp_output = nn.Dense(features=self.hidden_dim)(mlp_output)
            x = x + nn.Dropout(rate=self.dropout_rate, deterministic=not train)(mlp_output)

        # Classification head
        x = nn.LayerNorm()(x[:, 0])
        x = nn.Dense(features=self.num_classes)(x)

        return x


"""from flax import linen as nn
import jax.numpy as jnp
from jax import random
"""
class MBConvBlock(nn.Module):
    # Bloc MBConv utilisé dans EfficientNet
    expand_ratio: int  # Ratio d'expansion des canaux
    kernel_size: int   # Taille du noyau de convolution
    strides: int       # Strides pour la convolution
    filters: int       # Nombre de filtres de sortie
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool):
        input_filters = x.shape[-1]
        # Expansion phase
        if self.expand_ratio != 1:
            x = nn.Conv(features=input_filters * self.expand_ratio, kernel_size=(1, 1), padding='SAME')(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.relu(x)

        # Depthwise convolution
        x = nn.Conv(
            features=input_filters * self.expand_ratio,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.strides, self.strides),
            feature_group_count=input_filters * self.expand_ratio,
            padding='SAME'
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # Squeeze and excitation (optionnel, tu peux l'ignorer pour simplifier)
        # Ici, je le saute pour rester simple.

        # Projection phase
        x = nn.Conv(features=self.filters, kernel_size=(1, 1), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        # Skip connection
        #if x.shape == x.shape:  # Si les dimensions correspondent
        #    x = x + x  # Connexion résiduelle

        # Dropout
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        return x


"""class MBConvBlock(nn.Module):
    # Bloc MBConv utilisé dans EfficientNet
    expand_ratio: int  # Ratio d'expansion des canaux
    kernel_size: int   # Taille du noyau de convolution
    strides: int       # Strides pour la convolution
    filters: int       # Nombre de filtres de sortie
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool):
        input_filters = x.shape[-1]
        x_original = x  # Sauvegarder l'entrée originale pour la connexion résiduelle

        # Expansion phase
        if self.expand_ratio != 1:
            x = nn.Conv(features=input_filters * self.expand_ratio, kernel_size=(1, 1), padding='SAME')(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.swish(x)

        # Depthwise convolution
        x = nn.Conv(
            features=input_filters * self.expand_ratio,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.strides, self.strides),
            feature_group_count=input_filters * self.expand_ratio,
            padding='SAME'
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.swish(x)

        # Squeeze and excitation (optionnel, ajouté pour améliorer les performances)
        if self.expand_ratio != 1:
            squeeze = nn.Conv(features=input_filters * self.expand_ratio // 4, kernel_size=(1, 1), padding='SAME')(x)
            squeeze = nn.swish(squeeze)
            excitation = nn.Conv(features=input_filters * self.expand_ratio, kernel_size=(1, 1), padding='SAME')(squeeze)
            excitation = nn.sigmoid(excitation)
            x = x * excitation

        # Projection phase
        x = nn.Conv(features=self.filters, kernel_size=(1, 1), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        # Skip connection
        if input_filters == self.filters and self.strides == 1:
            x = x + x_original  # Connexion résiduelle

        # Dropout
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        return x
"""

"""class EfficientNetCIFAR10(nn.Module):
    num_classes: int
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x, train=False):
        # Première couche de convolution
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.swish(x)

        # Blocs MBConv
        x = MBConvBlock(expand_ratio=1, kernel_size=3, strides=1, filters=16)(x, train)
        x = MBConvBlock(expand_ratio=6, kernel_size=3, strides=2, filters=24)(x, train)
        x = MBConvBlock(expand_ratio=6, kernel_size=5, strides=1, filters=40)(x, train)
        x = MBConvBlock(expand_ratio=6, kernel_size=3, strides=2, filters=80)(x, train)
        x = MBConvBlock(expand_ratio=6, kernel_size=5, strides=1, filters=112)(x, train)
        x = MBConvBlock(expand_ratio=6, kernel_size=3, strides=2, filters=192)(x, train)

        # Dernière couche de convolution
        x = nn.Conv(features=1280, kernel_size=(1, 1), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.swish(x)

        # Pooling global
        x = nn.avg_pool(x, window_shape=(x.shape[1], x.shape[2]), strides=(1, 1))
        x = x.reshape((x.shape[0], -1))  # Flatten

        # Couche fully connected finale
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x"""


class EfficientNetB0(nn.Module):
    num_classes: int
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x, train=False):
        # Première couche de convolution
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        #x = nn.Conv(features=1024, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # Blocs MBConv (simplifié pour CIFAR-10)
        x = MBConvBlock(expand_ratio=1, kernel_size=3, strides=1, filters=16)(x, train)
        x = MBConvBlock(expand_ratio=6, kernel_size=3, strides=2, filters=24)(x, train)
        x = MBConvBlock(expand_ratio=6, kernel_size=5, strides=2, filters=40)(x, train)
        x = MBConvBlock(expand_ratio=6, kernel_size=3, strides=2, filters=80)(x, train)
        x = MBConvBlock(expand_ratio=6, kernel_size=5, strides=1, filters=112)(x, train)
        x = MBConvBlock(expand_ratio=6, kernel_size=5, strides=2, filters=192)(x, train)
        x = MBConvBlock(expand_ratio=6, kernel_size=3, strides=1, filters=320)(x, train)

        # Dernière couche de convolution
        x = nn.Conv(features=1280, kernel_size=(1, 1), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # Pooling global
        x = nn.avg_pool(x, window_shape=(x.shape[1], x.shape[2]), strides=(1, 1))
        x = x.reshape(x.shape[0], -1)  # Flatten

        # Couche fully connected finale
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x

class EfficientNetCIFAR10(nn.Module):
    num_classes: int
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x, train=False):
        # Première couche de convolution
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        # Première couche de convolution
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        # Blocs MBConv
        x = MBConvBlock(expand_ratio=1, kernel_size=3, strides=1, filters=16)(x, train)
        x = MBConvBlock(expand_ratio=6, kernel_size=3, strides=2, filters=24)(x, train)
        x = MBConvBlock(expand_ratio=6, kernel_size=5, strides=1, filters=40)(x, train)
        x = MBConvBlock(expand_ratio=6, kernel_size=3, strides=2, filters=80)(x, train)
        x = MBConvBlock(expand_ratio=6, kernel_size=5, strides=1, filters=112)(x, train)
        #x = MBConvBlock(expand_ratio=6, kernel_size=3, strides=2, filters=192)(x, train)

        # Dernière couche de convolution
        x = nn.Conv(features=1280, kernel_size=(1, 1), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # Pooling global
        #x = nn.avg_pool(x, window_shape=(x.shape[1], x.shape[2]), strides=(1, 1))
        x = nn.avg_pool(x, window_shape=(5, 5), strides=(1, 1))
        x = x.reshape(x.shape[0], -1)  # Flatten

        # Couche fully connected finale
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x
    
class simpleCNN(nn.Module):
    dropout_rate: float = 0.5  # Dropout à 0.5 par défaut
    use_se: bool = True
    num_classes: int = 47
    
    print("TEST")
    @nn.compact
    def __call__(self, x, train=True):
        # Convolution 1
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        #print(nn.Conv(features=32, kernel_size=(3, 3), padding="SAME"))
        x = nn.relu(x)
        # Convolution 2
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        # Flatten
        x = x.reshape(x.shape[0], -1)
        # Fully connected avec dropout
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x
