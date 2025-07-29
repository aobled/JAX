import jax
import os
import json
import gc
import jax.numpy as jnp
from flax import linen as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence
import sys

from JAX_ModelManager import ModelManager
from JAX_JsonModelsLibrary import build_model_from_json, visualize_network
from JAX_DataManager import load_cifar10_data, get_CIFAR10_like_dataset_size

from DepthwiseConv06 import DepthwiseConv06

# Configuration GPU
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

gc.collect()        # Force garbage collection
jax.clear_caches()  # Clear JAX caches
jax.devices()       # Reset the JAX devices

BATCH_SIZE = 128
EPOCH_NB = 1
LEARNING_RATE = 0.001
IMAGE_SIZE = 32
CHANNELS_NUM = 3
MODEL_NAME = 'depthwise_conv06_v2'
NUM_CLASSES = 10
REPORTING_DIR = './Graphe_Genomes/'

model_path = MODEL_NAME+".pth"
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
data_path = "./data/cifar-10-batches-py/"
#rng = jax.random.PRNGKey(0)


# 1. Chargement du modèle via configuration JSON
with open(MODEL_NAME+'.json', 'r') as file:
    data = json.load(file)
    json_config = data  # Pas besoin de json.dumps ici
model, config = build_model_from_json(json_config)
visualize_network(json_file=MODEL_NAME+'.json', output_filename=MODEL_NAME+'_graphviz')


# 2. chargement du dataset
data_dir = './data/cifar-10-batches-py/'
mean = jnp.array([0.4914, 0.4822, 0.4465])
std = jnp.array([0.2470, 0.2435, 0.2616])
train_dataset, test_dataset = load_cifar10_data(data_dir, mean, std)
dataset_size = get_CIFAR10_like_dataset_size(data_dir)

# 3. Initialisation du ModelManager
manager_json = ModelManager(model, config, dataset_size=dataset_size, model_name=MODEL_NAME, learning_rate=LEARNING_RATE, num_epochs=EPOCH_NB, batch_size=BATCH_SIZE)
manager_json.summarize_model()

# 4. Entraînement
manager_json.init_state()
manager_json.train_model(train_dataset, test_dataset, epochs=EPOCH_NB, batch_size=BATCH_SIZE)




# Remplacement en dur du model JSON par modèle statique
MODEL_NAME = 'depthwise_conv06_static'
model = DepthwiseConv06()
manager_class = ModelManager(model, config, dataset_size=dataset_size, model_name=MODEL_NAME, learning_rate=LEARNING_RATE, num_epochs=EPOCH_NB, batch_size=BATCH_SIZE)
manager_class.summarize_model()

# 4. Entraînement
manager_class.init_state()
manager_class.train_model(train_dataset, test_dataset, epochs=EPOCH_NB, batch_size=BATCH_SIZE)




"""
##### COMPARAISON: on commence par le modèle avec la classe statique
"""
variables_class = {
    "params": manager_class.state.params,
    "batch_stats": manager_class.batch_stats
}

input_sample = jnp.ones((1, 32, 32, 3))
rng = jax.random.PRNGKey(0)
rngs = {"dropout": rng}

# Modèle statique
_, summary_class = manager_class.model.apply(
    variables_class,
    input_sample,
    #config=None,
    config=manager_class.config,
    rng=rng,
    deterministic=True,
    mutable=["summary"]
)

"""
##### COMPARAISON: on continue avec le modèle avec la classe JSON
"""
variables_json = {
    "params": manager_json.state.params,
    "batch_stats": manager_json.batch_stats
}

input_sample = jnp.ones((1, 32, 32, 3))
rng = jax.random.PRNGKey(0)
rngs = {"dropout": rng}

# Modèle JSON dynamique
_, summary_json = manager_json.model.apply(
    variables_json,
    input_sample,
    config=manager_json.config,
    rng=rng,
    deterministic=True,
    mutable=["summary"]
)

for key in summary_json["summary"]:
    if key in summary_class["summary"]:
        act_json = summary_json["summary"][key][0]
        act_class = summary_class["summary"][key][0]
        mse = jnp.mean((act_json - act_class) ** 2)
        print(f"[{key}] MSE = {mse:.6f}")
    else:
        print(f"[{key}] manquant dans modèle statique")


# Compare les batch_stats des deux modèles
bs_class = manager_class.batch_stats
bs_json = manager_json.batch_stats

for key in bs_class:
    if key in bs_json:
        m1 = bs_class[key]["mean"]
        m2 = bs_json[key]["mean"]
        v1 = bs_class[key]["var"]
        v2 = bs_json[key]["var"]
        
        mean_diff = jnp.mean((m1 - m2) ** 2)
        var_diff = jnp.mean((v1 - v2) ** 2)

        print(f"[{key}] Mean MSE = {mean_diff:.6f}, Var MSE = {var_diff:.6f}")
    else:
        print(f"[{key}] manquant dans bs_json")

