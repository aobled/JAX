import jax.numpy as jnp
import numpy as np
from JAX_DataManager import (
    load_cifar10_data, get_CIFAR10_like_dataset_size,
    load_mnist_data, get_mnist_dataset_size,
    load_cifar100_data, get_CIFAR100_like_dataset_size, load_cifar100_class_names,
    load_fighterjet_data, get_npz_dataset_size
)

DATASET_CONFIGS = {
    "MNIST": {
        "mean": np.array([0.1307]),
        "std": np.array([0.3081]),
        "input_shape": (1, 28, 28, 1),
        "channels": 1,
        "num_classes": 10,
        "data_dir": "./data/MNIST",
        "label_names": [str(i) for i in range(10)],
        "load_function": load_mnist_data,
        "size_function": get_mnist_dataset_size,
        "model_name": 'mnist_depthwise_conv09',
    },
    "CIFAR10": {
        "mean": np.array([0.4914, 0.4822, 0.4465]),
        "std": np.array([0.2470, 0.2435, 0.2616]),
        "input_shape": (1, 32, 32, 3),
        "channels": 3,
        "num_classes": 10,
        "data_dir": "./data/cifar-10-batches-py/",
        "label_names": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ],
        "load_function": load_cifar10_data,
        "size_function": get_CIFAR10_like_dataset_size,
        "model_name": 'depthwise_conv_chatGPT04',
    },
    "CIFAR100": {
        "mean": np.array([0.5071, 0.4867, 0.4408]),
        "std": np.array([0.2675, 0.2565, 0.2761]),
        "input_shape": (1, 32, 32, 3),
        "channels": 3,
        "num_classes": 100,
        "data_dir": "./data/cifar-100-python/",
        "label_names": [f"class_{i}" for i in range(100)],
        "load_function": load_cifar100_data,
        "size_function": get_CIFAR100_like_dataset_size,
        "model_name": 'CIFAR100_depthwise_conv09',
    },
    "FIGHTERJET": {
        "mean": np.array([0.4852, 0.5285, 0.5696]),
        "std": np.array([0.2538, 0.2503, 0.2782]),
        "input_shape": (1, 128, 128, 3),
        "channels": 3,
        "num_classes": 8,
        "data_dir": "./data/FigtherJet/",
        "label_names": ["c17", "f14", "f15", "f16", "f22", "f35", "rafale", "typhoon"],
        "load_function": load_fighterjet_data,
        "size_function": get_npz_dataset_size,
        "model_name": 'depthwise_conv_chatGPT05',
    }
}

def get_dataset_config(name):
    name = name.upper()
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {name} non supporté.")
    return DATASET_CONFIGS[name]

def load_dataset(name):
    """
    Charge automatiquement le dataset spécifié avec ses paramètres.
    
    Args:
        name (str): Nom du dataset ("MNIST", "CIFAR10", etc.)
        
    Returns:
        tuple: (train_dataset, test_dataset, dataset_size)
    """
    cfg = get_dataset_config(name)
    
    # Chargement du dataset
    train_dataset, test_dataset = cfg["load_function"](cfg["data_dir"], cfg["mean"], cfg["std"])
    
    # Calcul de la taille
    if name == "FIGHTERJET":
        dataset_size = cfg["size_function"](cfg["data_dir"] + '/fighterjet_train.npz')
    else:
        dataset_size = cfg["size_function"](cfg["data_dir"])
    
    return train_dataset, test_dataset, dataset_size 