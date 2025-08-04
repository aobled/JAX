# ModelManager_PMAP_Compatible.py (fusionné complet avec reporting et évaluation multi-device)
import os
import yaml
import tqdm
import sys
import optax
import numpy as np
import pickle
import numpy as np

import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.training import train_state
from flax.serialization import to_bytes, from_bytes
from flax.core import freeze, unfreeze
from flax import traverse_util

from functools import partial
from typing import Any, Dict, Tuple

from JAX_Reporting import Reporting
from JAX_JsonModelsLibrary import build_model_from_json

# ========== UTILS MULTI-DEVICE (TPU/GPU) ==========
def replicate_tree(tree):
    return jax.device_put_replicated(tree, jax.devices())

def reshape_for_pmap(batch):
    n_devices = jax.device_count()
    return {k: v.reshape((n_devices, -1) + v.shape[1:]) for k, v in batch.items()}

def is_multi_device():
    return jax.device_count() > 1


def unreplicate(x):
    if isinstance(x, jnp.ndarray) and x.ndim >= 1 and jax.local_device_count() > 1:
        return x[0]
    elif isinstance(x, dict):
        return {k: unreplicate(v) for k, v in x.items()}
    elif isinstance(x, train_state.TrainState):
        return x.replace(
            step=unreplicate(x.step),
            params=unreplicate(x.params),
            opt_state=unreplicate(x.opt_state)
        )
    return x


class ModelManager:
    def __init__(self, model: nn.Module, config: Dict[str, Any], dataset_size: int, model_name: str,
                 learning_rate: float = 0.001, num_epochs: int = 5, batch_size: int = 128, num_classes=10,
                 input_shape=(1, 32, 32, 3), label_names=None, mean=None, std=None, gradient_accumulation_steps: int = 2):
        self.model = model
        self.config = config
        self.dataset_size = dataset_size
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.rng = jax.random.PRNGKey(42)
        self.label_names = label_names if label_names is not None else []
        self.mean = mean if mean is not None else []
        self.std = std if std is not None else []
        self.reporting = Reporting(label_names=self.label_names, mean=self.mean, std=self.std)
        self.label_smoothing = True # Par défaut, on active le label smoothing
        self.use_warmup_scheduler = False  # Active à partir du test 4
        
        # Gradient Accumulation
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.checkpoint_path = model_name + ".pth"
        self.reporting_path = model_name + ".yml"
        self.multi_device = is_multi_device()

        self.all_epochs = []
        self.all_losses = []
        self.all_lr = []
        self.all_val_correct = []
        self.all_train_correct = []
        self.epoch_start = 0
        self.best_accuracy = 0.0

        if os.path.isfile(self.reporting_path):
            self.load_reporting_from_yaml(self.reporting_path)
            self.epoch_start = len(self.all_lr)
            self.best_accuracy = max(self.all_val_correct)
            print("Reporting", self.reporting_path, "loaded. Starting from epoch", self.epoch_start)

        self.init_state()
    
        if self.multi_device:
            self.p_train_step = jax.pmap(self.train_step)
            self.p_compute_gradients = jax.pmap(self.compute_gradients)
        else:
            self.p_train_step = jax.jit(self.train_step)
            self.p_compute_gradients = jax.jit(self.compute_gradients)

    def create_optimizer(self):
        steps_per_epoch = self.dataset_size // self.batch_size

        if self.use_warmup_scheduler:
            # === Warmup + Cosine ===
            warmup_epochs = 5
            decay_epochs = self.num_epochs - warmup_epochs
            warmup_steps = warmup_epochs * steps_per_epoch
            decay_steps = decay_epochs * steps_per_epoch

            warmup_schedule = optax.linear_schedule(
                init_value=0.0,
                end_value=self.learning_rate,
                transition_steps=warmup_steps
            )

            decay_schedule = optax.cosine_decay_schedule(
                init_value=self.learning_rate,
                decay_steps=decay_steps,
                alpha=0.1  # LR final = 10% du LR initial
            )

            self.scheduler = optax.join_schedules(
                schedules=[warmup_schedule, decay_schedule],
                boundaries=[warmup_steps]
            )
        else:
            # === Cosine pure sans warmup ===
            total_steps = self.num_epochs * steps_per_epoch
            self.scheduler = optax.cosine_decay_schedule(
                init_value=self.learning_rate,
                decay_steps=total_steps,
                alpha=0.1
            )

        return optax.adamw(self.scheduler, weight_decay=1e-4)


    def init_state(self):
        rng, init_rng = jax.random.split(self.rng)
        dummy_input = jnp.ones(self.input_shape)
        variables = self.model.init(init_rng, dummy_input, self.config, rng=init_rng)

        total_steps = (self.num_epochs * self.dataset_size) // self.batch_size
        
        """self.scheduler = optax.piecewise_constant_schedule(
            init_value=self.learning_rate,
            boundaries_and_scales={int(total_steps * 0.5): 0.5, int(total_steps * 0.75): 0.1}
        )"""
        optimizer = self.create_optimizer()
        """# Plan d'apprentissage : Piecewise Constant Schedule >>> f(total_steps)
        self.scheduler = optax.piecewise_constant_schedule(
            init_value=self.learning_rate,
            boundaries_and_scales={	
                int(total_steps * 0.3): 0.5,
                int(total_steps * 0.6): 0.5,
                int(total_steps * 0.9): 0.2
            }
        )"""
    
        # Optimizer avec clipping + adamw + scheduler
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=self.scheduler, weight_decay=1e-4)
        )        
        
        state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=variables['params'],
            tx=optimizer
        )
        batch_stats = variables.get('batch_stats', {})

        if self.multi_device:
            self.state = replicate_tree(state)
            self.batch_stats = replicate_tree(batch_stats)
        else:
            self.state = state
            self.batch_stats = batch_stats

    def count_parameters(self, params):
        return sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda p: p.size if hasattr(p, 'size') else 0, params)))

    def compute_gradients(self, state, batch_stats, batch):
        def loss_fn(params):
            variables = {'params': params, 'batch_stats': batch_stats}
            logits, new_model_state = self.model.apply(
                variables, batch['image'], self.config, rng=self.rng, mutable=['batch_stats']
            )
            if self.label_smoothing:  # Booléen que tu peux ajouter dans ta config
                labels = self.smooth_labels(batch['label'], num_classes=logits.shape[-1], epsilon=0.1)
                loss = optax.softmax_cross_entropy(logits, labels).mean()
            else:
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['label']).mean()

            return loss, (logits, new_model_state['batch_stats'])

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (logits, new_batch_stats)), grads = grad_fn(state.params)
        metrics = {
            'loss': loss,
            'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label'])
        }
        return grads, new_batch_stats, metrics, logits

    def train_step(self, state, batch_stats, batch):
        def loss_fn(params):
            variables = {'params': params, 'batch_stats': batch_stats}
            logits, new_model_state = self.model.apply(
                variables, batch['image'], self.config, rng=self.rng, mutable=['batch_stats']
            )
            if self.label_smoothing:  # Booléen que tu peux ajouter dans ta config
                labels = self.smooth_labels(batch['label'], num_classes=logits.shape[-1], epsilon=0.1)
                loss = optax.softmax_cross_entropy(logits, labels).mean()
            else:
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['label']).mean()

            return loss, (logits, new_model_state['batch_stats'])

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (logits, new_batch_stats)), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = {
            'loss': loss,
            'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label'])
        }
        return state, new_batch_stats, metrics

    #@partial(jax.pmap if is_multi_device() else jax.jit, static_argnums=0)
    #def p_train_step(self, state, batch_stats, batch):
    #    return self.train_step(state, batch_stats, batch)
    
    def train_model(self, train_dataset: Dict[str, jnp.ndarray], test_dataset: Dict[str, jnp.ndarray], epochs: int = 10):
        # tentative de rechargement (optionnel)
        try:
            self.load_checkpoint(self.model_name+".pth")
            print("✅ Checkpoint chargé : entraînement repris.")
        except FileNotFoundError:
            print("ℹ️ Aucun checkpoint trouvé, entraînement initialisé.")
        
        n_devices = jax.device_count()
        print("Devices number = ", n_devices)
        batch_size = self.batch_size
        steps_per_epoch = train_dataset['image'].shape[0] // batch_size
        best_accuracy = 0.0

        for epoch in range(self.epoch_start, self.epoch_start + epochs):
            self.reporting = Reporting(nb_epochs=epoch)
            indices = np.random.permutation(train_dataset['image'].shape[0])
            train_dataset['image'] = np.asarray(train_dataset['image'])[indices]
            train_dataset['label'] = np.asarray(train_dataset['label'])[indices]

            # Plus besoin d'accumulateurs pour l'accuracy train
            
            # Gradient Accumulation
            if self.multi_device:
                # En mode multi-device, utiliser une approche différente
                for i in tqdm.trange(steps_per_epoch):
                    start = i * batch_size
                    end = start + batch_size
                    batch = {
                        'image': jnp.array(train_dataset['image'][start:end]),  # Conversion différée ici
                        'label': jnp.array(train_dataset['label'][start:end])
                    }
                    
                    # Data augmentation (uniquement sur les images)
                    self.rng, subkey = jax.random.split(self.rng)
                    batch['image'] = self.augment_batch(batch['image'], subkey)

                    batch = reshape_for_pmap(batch)
                    self.state, self.batch_stats, metrics = self.p_train_step(self.state, self.batch_stats, batch)

                    # Pas besoin d'accumuler l'accuracy train ici, on l'évalue proprement après
            else:
                # Mode single-device avec gradient accumulation
                grad_accum = None
                accum_step = 0
                
                for i in tqdm.trange(steps_per_epoch):
                    start = i * batch_size
                    end = start + batch_size
                    batch = {
                        'image': jnp.array(train_dataset['image'][start:end]),  # Conversion différée ici
                        'label': jnp.array(train_dataset['label'][start:end])
                    }
                    
                    # Data augmentation (uniquement sur les images)
                    self.rng, subkey = jax.random.split(self.rng)
                    batch['image'] = self.augment_batch(batch['image'], subkey)

                    # === 1. Compute gradients ===
                    grads, new_batch_stats, metrics, logits = self.p_compute_gradients(self.state, self.batch_stats, batch)

                    # === 2. Accumulate ===
                    if grad_accum is None:
                        grad_accum = grads
                    else:
                        grad_accum = jax.tree_util.tree_map(lambda a, b: a + b, grad_accum, grads)

                    accum_step += 1

                    # === 3. Apply if accumulation_steps atteint ===
                    if accum_step == self.gradient_accumulation_steps:
                        # Normalisation du gradient
                        grad_accum = jax.tree_util.tree_map(lambda x: x / self.gradient_accumulation_steps, grad_accum)
                    
                        self.state = self.state.apply_gradients(grads=grad_accum)
                        self.batch_stats = new_batch_stats
                        grad_accum = None
                        accum_step = 0

                    # Pas besoin d'accumuler l'accuracy train ici, on l'évalue proprement après

            val_acc = self.evaluate_model_fast(test_dataset, label_eval='val')
            # Évaluer le train set avec le même mode que le val set (deterministic=True)
            train_acc = self.evaluate_model_fast(train_dataset, label_eval='train')

            if self.multi_device:
                loss = metrics['loss'].mean()
            else:
                loss = float(metrics['loss'])

            self.all_epochs.append(epoch)
            self.all_losses.append(float(loss))

            #self.all_lr.append(float(self.scheduler(self.state.step if not self.multi_device else self.state[0].step)))
            # Tu répliques state sur 8 devices → tu obtiens un PyTree de TrainState, pas une liste.
            # Tu dois extraire la version "non-répliquée" via unreplicate(...)
            step = unreplicate(self.state).step
            self.all_lr.append(float(self.scheduler(step)))

            self.all_val_correct.append(float(val_acc))
            self.all_train_correct.append(float(train_acc))

            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, LR: {float(self.scheduler(step)):.6f}, Train: {train_acc:.2f}%, Val: {val_acc:.2f}%")

            if val_acc > best_accuracy:
                best_accuracy = val_acc
                self.save_checkpoint(self.model_name+'.pth')
                print(f"Nouveau meilleur modèle sauvegardé avec une précision de {best_accuracy:.2f}")

            self.reporting = self.reporting.replace(
                all_epochs=self.all_epochs,
                all_losses=self.all_losses,
                all_lr=self.all_lr,
                all_val_correct=self.all_val_correct,
                all_train_correct=self.all_train_correct,
                model_name=self.model_name,
                nb_neurones=self.count_parameters(unreplicate(self.state).params)
            )
            self.save_reporting_to_yaml(self.reporting_path)
            if len(self.all_epochs) > 1:
                self.reporting.visualize_training_results()

    def evaluate_model(self, evaluated_dataset: Dict[str, jnp.ndarray], batch_size: int = 128, label_eval=None):
        num_samples = evaluated_dataset['image'].shape[0]
        accuracies = []

        for i in tqdm.tqdm(range(0, num_samples, batch_size), desc="Evaluating " + label_eval):
            batch = {
                'image': evaluated_dataset['image'][i:i + batch_size],
                'label': evaluated_dataset['label'][i:i + batch_size]
            }
            variables = {'params': unreplicate(self.state).params, 'batch_stats': unreplicate(self.batch_stats)}
            logits = self.model.apply(
                variables,
                batch['image'],
                self.config,
                rng=self.rng,
                deterministic=True,
                mutable=False
            )
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch['label'])
            accuracies.append(accuracy)
        
        # DEBUG
        #print("PRED:", jnp.argmax(logits, axis=-1))
        #print("LABEL:", batch['label'])

        return jnp.mean(jnp.array(accuracies))

    def evaluate_model_fast(self, evaluated_dataset: Dict[str, jnp.ndarray], label_eval="val") -> float:
        """
        Évalue rapidement un sous-ensemble du dataset (10% si 'train', 100% sinon),
        en utilisant des batchs de taille self.batch_size.
        """
        import numpy as np  # pour le tirage aléatoire contrôlé
    
        variables = {
            'params': unreplicate(self.state).params,
            'batch_stats': unreplicate(self.batch_stats)
        }
    
        @jax.jit
        def apply_fn(images, variables):
            logits = self.model.apply(
                variables,
                images,
                self.config,
                rng=self.rng,
                deterministic=True,
                mutable=False
            )
            return logits
    
        images = evaluated_dataset['image']
        labels = evaluated_dataset['label']
    
        if label_eval == "train":
            # ⚠️ Ne garder que 10% du dataset d'entraînement, de façon déterministe
            num_total = len(images)
            num_subset = max(1, num_total // 10)
            np.random.seed(42)  # Fixe pour reproductibilité
            subset_indices = np.random.choice(num_total, num_subset, replace=False)
            images = images[subset_indices]
            labels = labels[subset_indices]
    
        batch_size = self.batch_size
        total_correct = 0
        total_seen = 0
    
        for i in tqdm.tqdm(range(0, len(images), batch_size), desc="Evaluating " + label_eval):
            batch_images = jnp.array(images[i:i+batch_size])
            batch_labels = jnp.array(labels[i:i+batch_size])
    
            logits = apply_fn(batch_images, variables)
            preds = jnp.argmax(logits, axis=-1)
            total_correct += jnp.sum(preds == batch_labels)
            total_seen += len(batch_labels)
    
        accuracy = total_correct / total_seen
        return float(accuracy * 100)


    """
    Qu'est-ce que le Label Smoothing ?
    Tu transformes les one-hot labels y = [0,0,1,0,...] en versions adoucies :
        y_smooth = y * (1 - ε) + ε / num_classes
    Exemple pour ε = 0.1 et 10 classes :
        → [0,0,1,0,...] devient [0.011, 0.011, 0.911, 0.011, ...]
    """
    def smooth_labels(self, labels: jnp.ndarray, num_classes: int, epsilon: float = 0.1) -> jnp.ndarray:
        """Applique label smoothing sur un batch de labels entiers"""
        onehot = jax.nn.one_hot(labels, num_classes)
        smoothed = onehot * (1 - epsilon) + epsilon / num_classes
        return smoothed


    def save_reporting_to_yaml(self, filepath: str):
        data = {
            'all_epochs': self.all_epochs,
            'all_losses': self.all_losses,
            'all_lr': self.all_lr,
            'all_val_correct': self.all_val_correct,
            'all_train_correct': self.all_train_correct,
            'model_name': self.model_name,
            'nb_neurones': self.count_parameters(unreplicate(self.state).params)
        }
        with open(filepath, 'w') as f:
            yaml.dump(data, f)

    def load_reporting_from_yaml(self, filepath: str):
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        self.all_epochs = data['all_epochs']
        self.all_losses = data['all_losses']
        self.all_lr = data['all_lr']
        self.all_val_correct = data['all_val_correct']
        self.all_train_correct = data['all_train_correct']
        self.model_name = data['model_name']


    def summarize_model(self, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape
        print(">->->->->->-", input_shape)
        dummy_input = jnp.zeros(input_shape)

        variables = {'params': unreplicate(self.state).params}
        if self.batch_stats:
            variables['batch_stats'] = unreplicate(self.batch_stats)

        _, summaries = self.model.apply(
            variables,
            dummy_input,
            self.config,
            rng=self.rng,
            deterministic=True,
            mutable=['summary']
        )

        print("\n=== Param Layers ===")
        #flat_params = flax.traverse_util.flatten_dict(unfreeze(variables['params']))
        flat_params = traverse_util.flatten_dict(unfreeze(variables['params']))
        for k, v in flat_params.items():
            print(f"{'/'.join(k)} → shape: {v.shape}")

        print("\n=== Activations (Ordered by model structure) ===")
        collected = summaries.get('summary', {})
        for layer in self.config['layers']:
            layer_id = layer['id']
            layer_type = layer['type']
            if layer_id in collected:
                acts = collected[layer_id]
                if not isinstance(acts, (list, tuple)):
                    acts = [acts]
                for idx, act in enumerate(acts):
                    shape = getattr(act, 'shape', None)
                    if shape is not None:
                        label = f"{layer_id}[{idx}]" if len(acts) > 1 else layer_id
                        print(f"{label} ({layer_type}) → shape: {shape}")
                    else:
                        print(f"{layer_id} ({layer_type}) → [non-array output: {type(act)}]")
            else:
                print(f"{layer_id} ({layer_type}) → [no activation collected]")

    def save_model_with_definition(self, model, params, filepath):
        model_data = {
            "model_class": model.__class__,
            #"model_args": {"dropout_rate": model.dropout_rate, "num_classes": model.num_classes},
            "params": params['params'],  # Sauvegarder seulement les paramètres du modèle
        }
        
        # Sauvegarder batch_stats seulement s'ils existent
        if 'batch_stats' in params:
            model_data["batch_stats"] = params['batch_stats']
    
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        #print(f"Modèle sauvegardé dans {filepath}")


    def load_model_with_definition(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
    
        model_class = model_data["model_class"]
        model_args = model_data["model_args"]
    
        model = model_class(**model_args)
        params = {"params": model_data["params"]}
    
        # Charger batch_stats seulement s'ils existent
        if "batch_stats" in model_data:
            params["batch_stats"] = model_data["batch_stats"]
    
        print(f"Modèle chargé depuis {filepath}")
        return model, params  # Retourner les params et batch_stats s'ils existent
    
    def save_checkpoint(self, filepath: str = None):
        filepath = filepath or self.checkpoint_path
    
        # Retire les réplicas si en multi-device
        state = unreplicate(self.state)
        batch_stats = unreplicate(self.batch_stats)
    
        model_data = {
            "model_class": self.model.__class__,
            #"model_args": {
            #    "dropout_rate": getattr(self.model, "dropout_rate", None),
            #    "num_classes": getattr(self.model, "num_classes", self.num_classes)
            #},
            "params": state.params,
            "opt_state": state.opt_state,
            "step": state.step,
            "batch_stats": batch_stats
        }
    
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
        print(f"✅ Modèle sauvegardé dans {filepath}")

    def load_checkpoint(self, filepath: str = None):
        """
        Charge un checkpoint de modèle à partir d'un fichier pickle.

        Cette fonction restaure l'état du modèle, les paramètres, l'état de l'optimiseur et les batch_stats
        à partir d'un fichier de sauvegarde (par défaut self.checkpoint_path ou le chemin fourni).
        Elle réinstancie également le modèle à partir de sa classe sauvegardée, et crée un nouvel optimiseur
        (l'état de l'optimiseur est restauré uniquement pour les poids et le step, pas pour le scheduler).
        Si le modèle a été entraîné en multi-device, les états sont répliqués sur tous les devices.

        Args:
            filepath (str, optionnel): Chemin du fichier de checkpoint à charger. Si None, utilise self.checkpoint_path.

        Effets de bord:
            - Met à jour self.model, self.state, self.batch_stats avec les valeurs chargées.
            - Affiche des informations sur la reprise et le learning rate.
        """
        filepath = filepath or self.checkpoint_path
    
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
    
        model_class = model_data["model_class"]
        model = model_class()
    
        tx = self.create_optimizer()
    
        # Crée un optimizer neuf, mais réinjecte les poids et step
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=model_data["params"],
            tx=tx
        ).replace(
            step=model_data.get("step", 0)
        )
    
        self.model = model
        self.state = replicate_tree(state) if self.multi_device else state
        self.batch_stats = replicate_tree(model_data["batch_stats"]) if self.multi_device else model_data["batch_stats"]
    
        #self.epoch_start = model_data.get("epoch", 0)
        print(f"🔁 Reprise à partir de l'époque {self.epoch_start}")
        print(f"✅ Modèle chargé depuis {filepath} (optimiseur neuf)")
        print(f"Restating Learning Rate at", self.learning_rate)



    # Data augmentation
    @partial(jax.jit, static_argnums=(0,))
    def augment_batch(self, images, rng):
        """
        Applique la data augmentation à un batch d'images.

        Pour chaque image du batch, applique la fonction augment_image avec une clé aléatoire différente.
        Utilise jax.vmap pour vectoriser l'opération sur tout le batch.

        Args:
            images (jnp.ndarray): Batch d'images à augmenter (shape: [batch, H, W, C]).
            rng (jax.random.PRNGKey): Clé aléatoire pour la génération des transformations.

        Returns:
            jnp.ndarray: Batch d'images augmentées (mêmes dimensions).
        """
        keys = jax.random.split(rng, len(images))
        augmented_images = jax.vmap(self.augment_image)(images, keys)
        return augmented_images

    def augment_image(self, image, key):
        """
        Applique une série de transformations d'augmentation à une image unique.

        Transformations appliquées :
            - Recadrage aléatoire avec padding
            - Flip horizontal aléatoire
            - (Optionnel) Rotation aléatoire
            - (Optionnel) Cutout

        Args:
            image (jnp.ndarray): Image à augmenter (shape: [H, W, C]).
            key (jax.random.PRNGKey): Clé aléatoire pour la génération des transformations.

        Returns:
            jnp.ndarray: Image transformée.
        """
        key1, key2, key3, key4 = jax.random.split(key, 4)
        image = random_crop_with_padding(image, padding=4, key=key1)
        image = horizontal_flip(image, key2)
        angle = jax.random.uniform(key3, minval=-0.1, maxval=0.1) * jnp.pi
        image = self.rotate(image, angle)
        image = cutout(image, size=4, key=key4)
        return image

    def rotate(self, image, angle):
        """
        Effectue une rotation de l'image autour de son centre.

        Args:
            image (jnp.ndarray): Image à faire pivoter (shape: [H, W, C]).
            angle (float): Angle de rotation en radians.

        Returns:
            jnp.ndarray: Image pivotée.
        """
        center = (image.shape[0] // 2, image.shape[1] // 2)
        y, x = jnp.indices(image.shape[:2])
        y = y - center[0]
        x = x - center[1]
        new_y = jnp.round(jnp.cos(angle) * y - jnp.sin(angle) * x + center[0]).astype(int)
        new_x = jnp.round(jnp.sin(angle) * y + jnp.cos(angle) * x + center[1]).astype(int)
        new_y = jnp.clip(new_y, 0, image.shape[0] - 1)
        new_x = jnp.clip(new_x, 0, image.shape[1] - 1)
        return image[new_y, new_x]

    def translate(self, image, translation):
        """
        Effectue une translation de l'image selon un vecteur donné.

        Args:
            image (jnp.ndarray): Image à translater (shape: [H, W, C]).
            translation (tuple/list/array): Vecteur de translation (dx, dy).

        Returns:
            jnp.ndarray: Image translatée.
        """
        matrix = jnp.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])
        coords = jnp.meshgrid(jnp.arange(image.shape[0]), jnp.arange(image.shape[1]), indexing='ij')
        coords = jnp.stack([coords[1], coords[0], jnp.ones(image.shape[:2])], axis=-1)
        transformed_coords = coords @ matrix.T
        transformed_coords = transformed_coords[:, :, :2]
        new_x = jnp.clip(transformed_coords[:, :, 0], 0, image.shape[1] - 1)
        new_y = jnp.clip(transformed_coords[:, :, 1], 0, image.shape[0] - 1)
        return image[new_y.astype(int), new_x.astype(int)]


def random_crop_with_padding(image, padding, key):
    """
    Effectue un recadrage aléatoire de l'image après padding.

    L'image est d'abord paddée (bordure réfléchie), puis un crop aléatoire de la taille d'origine est extrait.

    Args:
        image (jnp.ndarray): Image à recadrer (shape: [H, W, C]).
        padding (int): Taille du padding à appliquer sur chaque bord.
        key (jax.random.PRNGKey): Clé aléatoire pour le tirage du crop.

    Returns:
        jnp.ndarray: Image recadrée aléatoirement.
    """
    padded = jnp.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    h, w, c = image.shape
    key_y, key_x = jax.random.split(key)
    max_y = padded.shape[0] - h
    max_x = padded.shape[1] - w
    crop_y = jax.random.randint(key_y, (), 0, max_y + 1)
    crop_x = jax.random.randint(key_x, (), 0, max_x + 1)
    return jax.lax.dynamic_slice(padded, (crop_y, crop_x, 0), (h, w, c))

def cutout(image, size, key):
    """
    Applique la technique du cutout : masque un carré de taille donnée à une position aléatoire.

    Args:
        image (jnp.ndarray): Image à masquer (shape: [H, W, C]).
        size (int): Taille du carré à masquer.
        key (jax.random.PRNGKey): Clé aléatoire pour la position du masque.

    Returns:
        jnp.ndarray: Image avec une zone masquée à zéro.
    """
    h, w, c = image.shape
    key_y, key_x = jax.random.split(key)
    center_y = jax.random.randint(key_y, (), 0, h)
    center_x = jax.random.randint(key_x, (), 0, w)
    y1 = jnp.clip(center_y - size // 2, 0, h - size)
    x1 = jnp.clip(center_x - size // 2, 0, w - size)
    mask = jnp.ones_like(image)
    mask_patch = jnp.zeros((size, size, c), dtype=image.dtype)
    mask = jax.lax.dynamic_update_slice(mask, mask_patch, (y1, x1, 0))
    return image * mask

def horizontal_flip(image, key):
    """
    Effectue un flip horizontal aléatoire de l'image.

    Args:
        image (jnp.ndarray): Image à flipper (shape: [H, W, C]).
        key (jax.random.PRNGKey): Clé aléatoire pour décider du flip.

    Returns:
        jnp.ndarray: Image éventuellement retournée horizontalement.
    """
    do_flip = jax.random.bernoulli(key, 0.5)
    return jax.lax.cond(do_flip, lambda x: x[:, ::-1, :], lambda x: x, image)

def mixup_batch(self, images, labels, key, alpha=0.2):
    """
    Applique la technique du mixup sur un batch d'images et de labels one-hot.

    Mélange linéairement chaque image et son label avec une autre image/label du batch, selon un coefficient lambda tiré d'une loi Beta.

    Args:
        images (jnp.ndarray): Batch d'images (shape: [batch, H, W, C]).
        labels (jnp.ndarray): Batch de labels one-hot (shape: [batch, num_classes]).
        key (jax.random.PRNGKey): Clé aléatoire pour le tirage des coefficients et du shuffle.
        alpha (float): Paramètre de la loi Beta pour le mixup.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: (images mélangées, labels mélangés)
    """
    # Tirage des coefficients lambda ∼ Beta(alpha, alpha)
    key, subkey = jax.random.split(key)
    lam = jax.random.beta(subkey, alpha, alpha, (images.shape[0],))
    lam = jnp.maximum(lam, 1 - lam)  # symétrie pour éviter trop petits lam

    # Shuffle du batch pour combiner aléatoirement
    key, subkey = jax.random.split(key)
    indices = jax.random.permutation(subkey, images.shape[0])
    images2 = images[indices]
    labels2 = labels[indices]

    # Mise en forme pour broadcast
    lam_img = lam.reshape(-1, 1, 1, 1)
    lam_lbl = lam.reshape(-1, 1)

    # Interpolation
    mixed_images = lam_img * images + (1 - lam_img) * images2
    mixed_labels = lam_lbl * labels + (1 - lam_lbl) * labels2

    return mixed_images, mixed_labels
