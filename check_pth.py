import flax
from flax.serialization import from_bytes
import jax.numpy as jnp
import numpy as np

# Suppose que tu as accès à un objet self.state (ou tu peux en créer un vide avec la même structure)
dummy_state = {
    'params': ... ,         # un dict vide ou la structure de tes params
    'opt_state': ... ,      # idem pour opt_state
    'batch_stats': ... ,    # idem
    'step': 0,
    'epoch': 0,
    'model_config': {},     # ou None
}

# Tu peux remplir chaque champ par un dict vide ou une valeur par défaut.
# L'important est d'avoir les bonnes clés.

with open("depthwise_conv06.pth", "rb") as f:
    serialized_data = f.read()

checkpoint_data = from_bytes(dummy_state, serialized_data)
print(checkpoint_data.keys())

# Fonction récursive pour afficher la structure et les shapes
def print_shapes(d, prefix=''):
    if isinstance(d, dict):
        for k, v in d.items():
            print_shapes(v, prefix + '/' + k)
    elif isinstance(d, (np.ndarray, jnp.ndarray)):
        print(f"{prefix}: shape={d.shape}, dtype={d.dtype}")
    else:
        print(f"{prefix}: {type(d)}")

# Vérifie les principaux blocs
print("=== PARAMS ===")
print_shapes(checkpoint_data.get('params', {}))

print("\n=== BATCH_STATS ===")
print_shapes(checkpoint_data.get('batch_stats', {}))

print("\n=== OPT_STATE ===")
print_shapes(checkpoint_data.get('opt_state', {}))

print("\n=== AUTRES INFOS ===")
for k in checkpoint_data:
    if k not in ['params', 'batch_stats', 'opt_state']:
        print(f"{k}: {type(checkpoint_data[k])}")




def count_params(params):
    total = 0
    for k, v in params.items():
        if isinstance(v, dict):
            total += count_params(v)
        elif hasattr(v, "size"):
            total += v.size
    return total

# Exemple d'utilisation :
total_params = count_params(checkpoint_data['params'])
print(f"Nombre total de paramètres : {total_params}")



"""
2. Vérifier la correspondance avec le JSON
Pour vérifier que chaque couche de ton JSON est bien présente dans les paramètres sauvegardés, tu peux :
Lister les clés présentes dans params et les comparer aux IDs de ton JSON.
Pour chaque couche de type conv, dense, etc., vérifie que tu retrouves bien un bloc correspondant (ex: Conv_0, Dense_0, etc.) dans params.
"""
import json

# Charger le JSON du modèle
with open('depthwise_conv06.json', 'r') as f:
    model_json = json.load(f)

# Extraire les IDs des couches attendues
expected_layers = []
for layer in model_json['layers']:
    if layer['type'] in ['conv', 'dense', 'batch_norm']:
        # Pour Flax, les noms sont souvent Conv_0, Dense_0, BatchNorm_0, etc.
        # Adapte si tu utilises d'autres conventions de nommage.
        lname = layer['type'].capitalize() + "_" + layer['id'].split('_')[-1]
        expected_layers.append(lname)

# Lister les clés présentes dans les params
actual_layers = set(checkpoint_data['params'].keys())

print("Couches attendues dans le JSON :")
print(expected_layers)
print("\nCouches présentes dans les params :")
print(list(actual_layers))

# Vérifier les correspondances
missing = [l for l in expected_layers if l not in actual_layers]
if missing:
    print("\nCouches manquantes dans les params :", missing)
else:
    print("\nToutes les couches attendues sont présentes dans les params !")



"""
3. Vérifier les shapes
Pour chaque couche, tu peux afficher la shape des poids et des biais :
"""

for layer_name, weights in checkpoint_data['params'].items():
    print(f"=== {layer_name} ===")
    if isinstance(weights, dict):
        for param_name, param_value in weights.items():
            print(f"  {param_name}: shape={param_value.shape}, dtype={param_value.dtype}")
    else:
        print(f"  {type(weights)}")
