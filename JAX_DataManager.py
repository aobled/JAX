import os
import pickle
import jax.numpy as jnp
from typing import Dict, Tuple
import struct
from typing import Union
import gzip
import numpy as np
from pathlib import Path



def normalize_image(image: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    """Normalize an image using the given mean and standard deviation."""
    return (image - mean) / std

def load_fighterjet_npz(filepath: str, mean: jnp.ndarray, std: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    data = np.load(filepath)
    images = data["image"].astype(np.float32) / 255.0
    labels = data["label"].astype(np.int32)
    images = normalize_image(images, mean, std)
    return {"image": jnp.array(images), "label": jnp.array(labels)}

def load_fighterjet_data(data_dir: str, mean: jnp.ndarray, std: jnp.ndarray) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Charge le dataset FighterJet à partir de .npz compressés"""
    train_file = os.path.join(data_dir, "fighterjet_train.npz")
    val_file = os.path.join(data_dir, "fighterjet_val.npz")

    train_dataset = load_fighterjet_npz(train_file, mean, std)
    val_dataset = load_fighterjet_npz(val_file, mean, std)

    return train_dataset, val_dataset

def get_npz_dataset_size(npz_path: str) -> int:
    """Retourne le nombre d'images dans un fichier .npz FighterJet"""
    with np.load(npz_path) as data:
        return data["image"].shape[0]

def load_cifar10_batch(filepath: str, mean: jnp.ndarray, std: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load a single batch of CIFAR-10 data and normalize the images."""
    with open(filepath, 'rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
        images = data_dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
        images = normalize_image(images, mean, std)
        labels = jnp.array(data_dict[b'labels'])
    return jnp.array(images), labels

def load_cifar10_data(data_dir: str, mean: jnp.ndarray, std: jnp.ndarray) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Load the entire CIFAR-10 dataset."""
    train_batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_batch = 'test_batch'

    train_images, train_labels = [], []

    for batch in train_batches:
        filepath = os.path.join(data_dir, batch)
        images, labels = load_cifar10_batch(filepath, mean, std)
        train_images.append(images)
        train_labels.append(labels)

    train_images = jnp.concatenate(train_images, axis=0)
    train_labels = jnp.concatenate(train_labels, axis=0)

    test_filepath = os.path.join(data_dir, test_batch)
    test_images, test_labels = load_cifar10_batch(test_filepath, mean, std)

    train_dataset = {'image': train_images, 'label': train_labels}
    test_dataset = {'image': test_images, 'label': test_labels}

    return train_dataset, test_dataset

def get_mnist_like_dataset_size(images_path):
    """Retourne le nombre d'images dans un fichier MNIST-like .gz"""
    with gzip.open(images_path, 'rb') as f:
        magic_number, num_images = np.frombuffer(f.read(8), dtype=np.uint32).byteswap()  # Lire les 8 premiers octets
    dataset_size = num_images
    
    return dataset_size
    
def get_CIFAR10_like_dataset_size(dir_path):
    # Retourne la somme des images contenues dans tous les fichiers d'un répertoire CIFAR-10-like.
    if isinstance(dir_path, Path):
        dir_path = str(dir_path)
    
    if not dir_path or not os.path.exists(dir_path):
        raise FileNotFoundError(f"The directory {dir_path} does not exist.")

    dataset_size = 0
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path) and file_name.startswith("data_batch_"):
            try:
                with open(file_path, 'rb') as f:
                    batch_dict = pickle.load(f, encoding='bytes')
                    num_images = len(batch_dict[b'data'])
                    dataset_size += num_images
                    #print(f"{file_name}: {num_images} images")
            except Exception as e:
                print(f"Erreur lors du traitement de {file_name}: {e}")

    print("Total dataset_size=", dataset_size)
    
    return dataset_size


def read_idx_images(path: str) -> jnp.ndarray:
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        images = images[..., np.newaxis]  # (num, 28, 28, 1)
        return jnp.array(images, dtype=jnp.float32) / 255.0

def read_idx_labels(path: str) -> jnp.ndarray:
    with gzip.open(path, 'rb') as f:
        magic, num = np.frombuffer(f.read(8), dtype='>i4')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return jnp.array(labels)

def load_mnist_data(data_dir: str, mean: jnp.ndarray = 0.1307, std: jnp.ndarray = 0.3081) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Load the MNIST dataset and normalize images."""
    train_images = read_idx_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    train_labels = read_idx_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    
    test_images = read_idx_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    test_labels = read_idx_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))

    # Normalize
    train_images = normalize_image(train_images, mean, std)
    test_images = normalize_image(test_images, mean, std)

    train_dataset = {'image': train_images, 'label': train_labels}
    test_dataset = {'image': test_images, 'label': test_labels}

    return train_dataset, test_dataset


def get_mnist_dataset_size(dir_path: Union[str, Path]) -> int:
    """
    Calcule la taille totale (nombre d'images) des fichiers MNIST présents dans un répertoire donné.
    Prend en compte uniquement les fichiers 'train-images' et 't10k-images'.
    """
    if isinstance(dir_path, Path):
        dir_path = str(dir_path)

    if not dir_path or not os.path.exists(dir_path):
        raise FileNotFoundError(f"The directory {dir_path} does not exist.")

    total_size = 0
    for file_name in os.listdir(dir_path):
        if file_name.endswith("-images-idx3-ubyte.gz"):
            file_path = os.path.join(dir_path, file_name)
            try:
                with gzip.open(file_path, 'rb') as f:
                    magic, num_images = struct.unpack(">II", f.read(8))
                    print(f"{file_name}: {num_images} images")
                    total_size += num_images
            except Exception as e:
                print(f"Erreur lors du traitement de {file_name}: {e}")

    print("Total dataset size =", total_size)
    return total_size



def load_cifar100_batch(filepath: str, mean: jnp.ndarray, std: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Charge un batch de CIFAR-100 et applique la normalisation."""
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        # Les images sont stockées sous b'data', labels sous b'fine_labels'
        images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
        labels = jnp.array(batch[b'fine_labels'])  # labels "fin" (100 classes)
        images = normalize_image(images, mean, std)
        return jnp.array(images), labels

def load_cifar100_data(data_dir: str, mean: jnp.ndarray, std: jnp.ndarray) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Charge CIFAR-100 depuis les fichiers 'train' et 'test'."""
    train_file = os.path.join(data_dir, 'train')
    test_file = os.path.join(data_dir, 'test')

    train_images, train_labels = load_cifar100_batch(train_file, mean, std)
    test_images, test_labels = load_cifar100_batch(test_file, mean, std)

    train_dataset = {'image': train_images, 'label': train_labels}
    test_dataset = {'image': test_images, 'label': test_labels}
    
    return train_dataset, test_dataset

def load_cifar100_class_names(meta_file: str) -> list:
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
        return [name.decode('utf-8') for name in meta[b'fine_label_names']]



def get_CIFAR100_like_dataset_size(dir_path):
    """
    Retourne le nombre total d'images contenues dans les fichiers CIFAR-100 'train' et 'test'.
    """
    if isinstance(dir_path, Path):
        dir_path = str(dir_path)

    if not dir_path or not os.path.exists(dir_path):
        raise FileNotFoundError(f"The directory {dir_path} does not exist.")

    dataset_size = 0
    for file_name in ["train", "test"]:
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'rb') as f:
                    data_dict = pickle.load(f, encoding='bytes')
                    num_images = len(data_dict[b'data'])
                    dataset_size += num_images
                    print(f"{file_name}: {num_images} images")
            except Exception as e:
                print(f"Erreur lors du traitement de {file_name}: {e}")

    print("Total dataset_size =", dataset_size)
    return dataset_size
