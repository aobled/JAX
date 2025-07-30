import jax
import os
import json
import gc
import jax.numpy as jnp
import sys

from JAX_ModelManager import ModelManager
from JAX_JsonModelsLibrary import build_model_from_json, visualize_network
from JAX_DatasetConfigs import get_dataset_config, load_dataset


BATCH_SIZE = 64
EPOCH_NB = 50
LEARNING_RATE = 0.001
REPORTING_DIR = './Graphe_Genomes/'

# Choix du dataset cible
DATASET_NAME = "CIFAR10"  # "CIFAR10"  "MNIST", "CIFAR100", "FIGHTERJET"
MODEL_NAME = 'depthwise_conv_chatGPT04'
# Chargement automatique du dataset
cfg = get_dataset_config(DATASET_NAME)
train_dataset, test_dataset, dataset_size = load_dataset(DATASET_NAME)

# Chargement du modèle via configuration JSON
with open(MODEL_NAME+'.json', 'r') as file:
    data = json.load(file)
    json_config = data  # Pas besoin de json.dumps ici
model, config = build_model_from_json(json_config)
visualize_network(json_file=MODEL_NAME+'.json', output_filename=MODEL_NAME+'_graphviz')


# 3. Initialisation du ModelManager
manager = ModelManager(model,
                       config,
                       dataset_size=dataset_size,
                       model_name=MODEL_NAME,
                       learning_rate=LEARNING_RATE, 
                       num_epochs=EPOCH_NB, 
                       batch_size=BATCH_SIZE,
                       num_classes=cfg["num_classes"],
                       input_shape=cfg["input_shape"],
                       label_names=cfg["label_names"],
                       mean=cfg["mean"],
                       std=cfg["std"], 
                       gradient_accumulation_steps=2)

manager.summarize_model()

# 4. Entraînement
print("launch TRAINING GPU L4 1 batch de 128")
manager.train_model(train_dataset, test_dataset, epochs=EPOCH_NB)

# Affichage des erreurs du modèle sauvegardé sur le test_dataset
manager.reporting.show_errors_from_pth(dataset=test_dataset,
                                       mean=cfg["mean"], 
                                       std=cfg["std"],
                                       pth_path=MODEL_NAME+".pth", 
                                       json_path=MODEL_NAME+".json", 
                                       err_png_path=MODEL_NAME+"_errors.png", 
                                       max_errors=9)
"""import pickle
import matplotlib.pyplot as plt
import numpy as np

# Charger le modèle à partir du fichier .pth
def load_model(filepath):
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

# Visualiser les cartes d'activation
def visualize_activation_maps(model, input_data):
    # Assurez-vous que votre modèle est configuré pour renvoyer les activations
    # Passez les données d'entrée à travers le modèle pour obtenir les activations
    activations = model.predict(input_data)  # Remplacez par la méthode appropriée pour obtenir les activations

    # Visualiser les cartes d'activation
    for i, activation_map in enumerate(activations):
        plt.imshow(activation_map, cmap='viridis')
        plt.title(f'Activation Map {i+1}')
        plt.colorbar()
        plt.show()

# Exemple d'utilisation
filepath = 'depthwise_conv_chatGPT04.pth'
model_data = load_model(filepath)

# Créez une instance de votre modèle et chargez les paramètres
# model = YourModelClass()  # Remplacez par votre classe de modèle
#model.load_params(model_data['params'])

# Générez ou chargez des données d'entrée
input_data = np.random.rand(1, 32, 32, 3)  # Exemple de données aléatoires

# Visualisez les cartes d'activation
visualize_activation_maps(manager.model, input_data)

"""