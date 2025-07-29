import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Dict
import json
import torch.nn.functional as F
from typing import Optional

import sys
import os

class ResidualAdd(nn.Module):
    out_channels: int
    activation: Optional[str] = None  # ex. "relu", "silu", ...

    @nn.compact
    def __call__(self, *inputs):
        assert len(inputs) >= 2, "ResidualAddN nécessite au moins 2 entrées"

        # On prend la 1ère entrée comme référence
        ref = inputs[0]
        ref_shape = ref.shape
        aligned = []

        for i, x in enumerate(inputs):
            # Alignement spatial si nécessaire
            if x.shape[1:3] != ref_shape[1:3]:
                x = jax.image.resize(x, shape=ref_shape, method='nearest')

            # Alignement des canaux si nécessaire
            if x.shape[-1] != self.out_channels:
                x = nn.Conv(self.out_channels, kernel_size=(1, 1), padding='SAME', use_bias=False)(x)

            aligned.append(x)

        out = sum(aligned)

        if self.activation is not None:
            try:
                out = getattr(nn, self.activation)(out)
            except AttributeError:
                raise ValueError(f"Activation inconnue : {self.activation}")

        return out

"""class AdaptiveResidualAdd(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x, residual):
        # Resize spatialement si nécessaire
        if x.shape[1:3] != residual.shape[1:3]:
            residual = jax.image.resize(residual, x.shape[:3] + (residual.shape[-1],), method='bilinear')

        # Adapter les canaux si nécessaire
        if residual.shape[-1] != self.out_channels:
            residual = nn.Conv(self.out_channels, kernel_size=(1, 1), use_bias=False)(residual)
            #residual = nn.BatchNorm(use_running_average=False)(residual)

        # Appliquer un coefficient d'ajustement learnable
        alpha = self.param("alpha", lambda key: jnp.ones((1, 1, 1, 1)))
        return x + alpha * residual"""

"""class ResidualAdd(nn.Module):
    in_channels: int
    out_channels: int

    @nn.compact
    def __call__(self, x, residual):
        # Adapter spatialement si nécessaire
        if x.shape[1:3] != residual.shape[1:3]:
            residual = jax.image.resize(residual, x.shape, method='nearest')

        # Adapter les canaux avec 1x1 conv si nécessaire
        if self.in_channels != self.out_channels:
            residual = nn.Conv(features=self.out_channels, kernel_size=(1, 1))(residual)

        return x + residual"""
"""class ResidualAdd(nn.Module):
    in_channels: int
    out_channels: int
    activation: Optional[str] = None  # ex. "relu", "silu", ...

    @nn.compact
    def __call__(self, x, residual):
        # Adapter spatialement
        if x.shape[1:3] != residual.shape[1:3]:
            residual = jax.image.resize(residual, shape=x.shape, method='nearest')

        # Adapter les canaux
        if self.in_channels != self.out_channels:
            residual = nn.Conv(features=self.out_channels, kernel_size=(1, 1), padding='SAME', use_bias=False)(residual)
            #residual = nn.BatchNorm(use_running_average=False)(residual)

        out = x + residual

        # Appliquer l'activation dynamiquement si spécifiée
        if self.activation is not None:
            try:
                out = getattr(nn, self.activation)(out)
            except AttributeError:
                raise ValueError(f"Activation inconnue : {self.activation}")

        return out"""
    
class AdjustShapeAndChannels(nn.Module):
    target_shape: tuple  # (B, H, W, C)

    @nn.compact
    def __call__(self, x):
        B, H_t, W_t, C_t = self.target_shape
        H_x, W_x, C_x = x.shape[1:]

        # Resize spatial if needed
        if (H_x, W_x) != (H_t, W_t):
            x = jax.image.resize(x, (B, H_t, W_t, C_x), method='bilinear')

        # Adjust channels if needed
        if C_x != C_t:
            x = nn.Conv(C_t, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(x)

        return x


class ResidualProjection(nn.Module):
    target_shape: tuple  # (B, H, W, C)

    @nn.compact
    def __call__(self, x, deterministic=False):
        _, H_t, W_t, C_t = self.target_shape
        B, H_x, W_x, C_x = x.shape

        needs_conv = (H_x != H_t or W_x != W_t or C_x != C_t)
        if needs_conv:
            stride_h = H_x // H_t if H_x != H_t else 1
            stride_w = W_x // W_t if W_x != W_t else 1
            x = nn.Conv(
                C_t, kernel_size=(1, 1),
                strides=(stride_h, stride_w),
                use_bias=False,
                name='res_proj_conv'
            )(x)
            #x = nn.BatchNorm(use_running_average=False, name='res_proj_bn')(x)
            x = nn.BatchNorm(use_running_average=deterministic)(x)

        return x


class DynamicModel(nn.Module):    
    @nn.compact
    def __call__(self, x, config, rng, deterministic=False):
        #outputs = {'input': x}
        outputs = {}
        #print("######## DEBUT")

        # Traverse the layers and build the model
        for layer in config['layers']:

            if layer['type'] == 'input':
                out = x
            elif layer['type'] == 'dense':
                out = nn.Dense(features=layer['units'])(outputs[layer['inputs'][0]])
            elif layer['type'] == 'conv':                
                in_channels = outputs[layer['inputs'][0]].shape[-1]
                groups = layer.get("groups", 1)
                if in_channels % groups != 0:
                    raise ValueError(f"Invalid conv groups: {groups} for in_channels: {in_channels}")
                
                out = nn.Conv(
                    features=layer['filters'],
                    kernel_size=tuple(layer['kernel_size']),
                    strides=tuple(layer.get('strides', (1, 1))),
                    padding=layer.get('padding', 'VALID'),
                    feature_group_count=layer.get("groups", 1)
                )(outputs[layer['inputs'][0]])
                #print("######## conv.out.shape=", out.shape)
            elif layer['type'] == 'relu':
                out = nn.relu(outputs[layer['inputs'][0]])
            elif layer['type'] == 'gelu':
                out = nn.gelu(outputs[layer['inputs'][0]])
            elif layer['type'] == 'silu':
                out = nn.silu(outputs[layer['inputs'][0]])
            elif layer['type'] == 'tanh':
                out = nn.tanh(outputs[layer['inputs'][0]])
            elif layer['type'] == 'sigmoid':
                out = nn.sigmoid(outputs[layer['inputs'][0]])
            elif layer['type'] == 'leaky_relu':
                out = nn.leaky_relu(outputs[layer['inputs'][0]], negative_slope=0.01)
            elif layer['type'] == 'softplus':
                out = nn.softplus(outputs[layer['inputs'][0]])
            elif layer['type'] == 'elu':
                out = nn.elu(outputs[layer['inputs'][0]])
            elif layer['type'] == 'selu':
                out = nn.selu(outputs[layer['inputs'][0]])
            elif layer['type'] == 'layer_norm':
                out = nn.LayerNorm(epsilon=1e-5)(outputs[layer['inputs'][0]])
            elif layer['type'] == 'flatten':
                out = outputs[layer['inputs'][0]].reshape((outputs[layer['inputs'][0]].shape[0], -1))
            elif layer['type'] == 'avg_pool':
                out = nn.avg_pool(
                    outputs[layer['inputs'][0]],
                    window_shape=tuple(layer['window_shape']),
                    strides=tuple(layer.get('strides', (1, 1))),
                    padding=layer.get('padding', 'VALID')
                )
            elif layer['type'] == 'batch_norm':
                #Dans Flax :
                #    - use_running_average=True -> mode inference (on n'update pas les stats)
                #    - use_running_average=False -> mode training (on met à jour les moyennes/variances)
                batch_norm_layer = nn.BatchNorm(
                    momentum=layer.get('momentum', 0.99),
                    epsilon=layer.get('epsilon', 1e-5),
                    use_bias=False,
                    use_scale=False,
                    bias_init=nn.initializers.zeros,
                    scale_init=nn.initializers.ones
                )
                out = batch_norm_layer(outputs[layer['inputs'][0]], use_running_average=deterministic)
            elif layer['type'] == 'max_pool':
                out = nn.max_pool(
                    outputs[layer['inputs'][0]],
                    window_shape=tuple(layer.get('window_shape', (2, 2))),
                    strides=tuple(layer.get('strides', (2, 2))),
                    padding=layer.get('padding', 'VALID')
                )
            elif layer['type'] == 'global_max_pool':
                out = jnp.max(
                    outputs[layer['inputs'][0]],
                    axis=(1, 2),
                    keepdims=layer.get('keepdims', False)
                )
            elif layer['type'] == 'global_avg_pool': # Assuming the spatial dimensions are at axes 1 and 2
                out = jnp.mean(outputs[layer['inputs'][0]], 
                               axis=(1, 2),
                               #keepdims=layer['keepdims'])
                               keepdims=layer.get('keepdims', False)
                              )
            elif layer['type'] == 'dropout':
                rng, subkey = jax.random.split(rng)  # Split the RNG key for dropout
                out = nn.Dropout(rate=layer['rate'])(outputs[layer['inputs'][0]], deterministic=deterministic, rng=subkey)
            
            #elif layer['type'] == 'add':
            #    x1 = outputs[layer['inputs'][0]]
            #    x2 = outputs[layer['inputs'][1]]
            #    in_ch = x2.shape[-1]
            #    out_ch = x1.shape[-1]
            #    act = layer.get("activation", None)  # récupère "silu", "relu", etc.
            #    out = ResidualAdd(
            #        in_channels=in_ch,
            #        out_channels=out_ch,
            #        activation=act
            #    )(x1, x2)
            elif layer['type'] == 'add':
                inputs_tensors = [outputs[i] for i in layer['inputs']]
                out_ch = inputs_tensors[0].shape[-1]  # tu peux aussi faire un max ou une moyenne selon ton cas
                act = layer.get("activation", None) # récupère "silu", "relu", etc.
                
                out = ResidualAdd(out_channels=out_ch, activation=act)(*inputs_tensors)
                
            elif layer["type"] == "concat":
                # Par défaut, on concatène sur l'axe des canaux (-1)
                axis = layer.get("axis", -1)
                tensors = [outputs[inp] for inp in layer["inputs"]]
                out = jnp.concatenate(tensors, axis=axis)
            
            elif layer['type'] == 'reshape':
                input_tensor = outputs[layer['inputs'][0]]
                new_shape = tuple(layer['shape'])
                out = jnp.reshape(input_tensor, new_shape)
            elif layer['type'] == 'multiply':
                x1 = outputs[layer['inputs'][0]]
                x2 = outputs[layer['inputs'][1]]
                out = x1 * x2
                # injection activation si nécessaire
                act = layer.get("activation", None)
                if act is not None:
                    out = getattr(nn, act)(out)  # appelle nn.silu, nn.relu...

            elif layer['type'] == 'broadcast':
                input_tensor = outputs[layer['inputs'][0]]
                # broadcast : (N, C) → (N, 1, 1, C)
                out = input_tensor[:, None, None, :]

            elif layer['type'] == 'se_block':
                #fait référence à un Squeeze-and-Excitation block (SE Block), une attention canal légère, 
                # popularisée dans SENet et utilisée dans EfficientNet / MobileNetV3.
                # peut être remplacé par des couches natives donc:
                #   - GAP sur l'input_tensor
                #   - puis Dense(size input//8)
                #   - puis silu
                #   - puis Dense(size input)
                #   - puis sigmoid
                #   - puis un nouveau layer de type broadcast ???
                #   - et enfin un multiply de tout ca avec input_tensor 
                # DONC A VIRER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                input_tensor = outputs[layer['inputs'][0]]
                channels = input_tensor.shape[-1]
                reduction = layer.get('reduction', 8)
            
                # Squeeze
                se = jnp.mean(input_tensor, axis=(1, 2))  # shape (N, C)
            
                # Excitation
                se = nn.Dense(channels // reduction)(se)
                se = nn.silu(se)
                se = nn.Dense(channels)(se)
                se = nn.sigmoid(se)
                se = se[:, None, None, :]  # broadcast
            
                # Scale
                out = input_tensor * se
            else:
                raise ValueError(f"Layer type '{layer['type']}' is not implemented!")

            self.sow('summary', layer['id'], out)
            outputs[layer['id']] = out
            #print(">>>>>>>>>>> [", layer['id'], "]", layer['type'], outputs[layer['id']].shape)
            
        # Return the final output
        #sys.exit()
        return outputs[config['output_layer']]


class LayerConfig:
    """Configuration class for each layer in the model."""

    def __init__(self, layer_type: str, **kwargs):
        self.layer_type = layer_type
        self.kwargs = kwargs

def build_model_from_json(json_config: Dict[str, Any]) -> nn.Module:
    """Builds a DynamicModel from a JSON configuration."""
    model = DynamicModel()
    return model, json_config  # Return the model and the config



import networkx as nx
import matplotlib.pyplot as plt

def build_graph_from_json(json_config):
    G = nx.DiGraph()
    node_labels = {}

    for layer in json_config["layers"]:
        node_id = layer["id"]
        layer_type = layer["type"]
        label = f"{layer_type.upper()}"
        if layer_type == "conv":
            label += f"\n{layer.get('filters', '?')}@{layer.get('kernel_size', '?')}"
        elif layer_type == "dense":
            label += f"\n{layer.get('units', '?')}"
        elif layer_type == "add":
            label = "+"
        elif layer_type == "multiply":
            label = "*"
        elif layer_type == "concat":
            label = "CONCAT"
        elif layer_type == "max_pool":
            label = "MaxPool"
        elif layer_type == "avg_pool":
            label = "AvgPool"
        elif layer_type == "global_avg_pool":
            label = "GlobalAvg"
        elif layer_type == "input":
            label = "Input"
        elif layer_type == "output":
            label = "Output"

        G.add_node(node_id)
        node_labels[node_id] = label

        for input_id in layer.get("inputs", []):
            G.add_edge(input_id, node_id)

    return G, node_labels

"""# Utilisation avec ResNet32_CIFAR10.json
G, node_labels = build_graph_from_json(json_config)

# Tracé avec graphviz_layout si disponible
try:
    from networkx.drawing.nx_agraph import graphviz_layout
    pos = graphviz_layout(G, prog="dot")
except ImportError:
    pos = nx.spring_layout(G)
"""

def plot_json_graph(json_config, title="Architecture Graph"):
    G, node_labels = build_graph_from_json(json_config)

    # Try Graphviz layout first (top-down)
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog="dot")  # vertical
    except ImportError:
        print("⚠️ Graphviz layout not available, falling back to spring layout.")
        pos = nx.spring_layout(G)

    plt.figure(figsize=(40, 24))
    nx.draw(G, pos,
            with_labels=True,
            labels=node_labels,
            node_size=2500,
            node_color="lightblue",
            edge_color="gray",
            font_size=10,
            font_weight="bold",
            arrows=True)
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()



from graphviz import Digraph

def visualize_network(json_file, output_filename='network'):
    # Charger le JSON
    with open(json_file) as f:
        data = json.load(f)
    
    # Créer un graphe orienté
    dot = Digraph(comment='Network Architecture', format='png')
    dot.attr(rankdir='TB')  # De haut en bas (Top to Bottom)
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightgrey')
    
    # Dictionnaire pour suivre les connexions déjà tracées
    connections = set()
    
    # Ajouter toutes les couches comme noeuds
    for layer in data['layers']:
        layer_id = layer['id']
        layer_type = layer['type']
        dot.node(layer_id, f"{layer_id} ({layer_type})")
    
    # Ajouter les connexions
    for layer in data['layers']:
        for input_id in layer.get('inputs', []):
            if (input_id, layer['id']) not in connections:
                dot.edge(input_id, layer['id'])
                connections.add((input_id, layer['id']))
    
    # Enregistrer le rendu PG et supprimer 
    dot.render(output_filename, view=False)
    os.remove(output_filename)
    

#TEST
"""# Charger la configuration JSON
with open('tmp.json', 'r') as file:
    data = json.load(file)
    json_config = data  # Pas besoin de json.dumps ici

# Construire le modèle et obtenir la configuration
model, config = build_model_from_json(json_config)

# Initialiser les variables du modèle en passant la configuration
#variables = model.init(jax.random.PRNGKey(0), jnp.ones([1, 32, 32, 3]), config)  # Example input

print(config)
visualize_network(json_file='tmp.json', output_filename='tmp_graphviz')"""


