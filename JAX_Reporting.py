from flax import struct
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from scipy.integrate import simpson
import numpy as np
import jax.numpy as jnp
import pickle, json
from JAX_JsonModelsLibrary import build_model_from_json
import jax
import tqdm


@struct.dataclass
class Reporting:
    nb_epochs: int = 0
    all_epochs: list = struct.field(default_factory=list)
    all_losses: list = struct.field(default_factory=list)
    all_lr: list = struct.field(default_factory=list)
    all_val_correct: list = struct.field(default_factory=list)
    all_train_correct: list = struct.field(default_factory=list)
    model_name: str = ''
    test_correct: float = 0.0
    nb_neurones: int = 0
    label_names: list = struct.field(default_factory=list)
    mean: list = struct.field(default_factory=list)
    std: list = struct.field(default_factory=list)
    
    def visualize_training_results(self, epoch_start=0):
        plt.rc('mathtext', default='regular')
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot()
        ax2 = ax.twinx()
    
        # Plot loss and accuracy
        ax.plot('epochs', 'losses', data={'epochs': self.all_epochs[epoch_start:], 'losses': self.all_losses[epoch_start:]}, label='Epoch', color='tab:orange')
        ax2.plot('epochs', 'train_correct', data={'epochs': self.all_epochs[epoch_start:], 'train_correct': self.all_train_correct[epoch_start:]}, color='tab:red', label='Train', linewidth=2, alpha=0.15)
        ax2.fill_between(self.all_epochs[epoch_start:], self.all_train_correct[epoch_start:], self.all_val_correct[epoch_start:], color='tab:red', alpha=0.15)
        ax2.plot('epochs', 'val_correct', '-', data={'epochs': self.all_epochs[epoch_start:], 'val_correct': self.all_val_correct[epoch_start:]}, color='tab:blue', label='Val')
    
        # Plot learning rate as a line
        ax3 = ax.twinx()
        ax3.set_yscale('log')  # Set logarithmic scale for learning rate
        ax3.plot(self.all_epochs[epoch_start:], self.all_lr[epoch_start:], color='lightgreen', label='LR')
        ax3.set_ylabel('LR', color='lightgreen')
        #ax3.set_ylim(bottom=0.00001, top=0.001)  # Adjust these values as needed
        ax3.set_ylim(bottom=min(self.all_lr[epoch_start:])/1.01, top=max(self.all_lr[epoch_start:])*1.01)  # Adjust these values as needed
        ax3.yaxis.set_major_locator(LogLocator(base=10.0))
        
        # Set tick colors for each axis
        ax.tick_params(axis='y', colors='tab:orange')
        ax2.tick_params(axis='y', colors='tab:blue')
        ax3.tick_params(axis='y', colors='tab:green')
                
        ax2.legend(loc=9)
        ax.grid()
        ax.set_xlabel("Epochs")
        ax.set_ylabel(f'Loss (min: {round(self.all_losses[-1], 5)})', color='tab:orange')
        ax2.set_ylabel('Accuracy %', color='tab:blue')
    
        # Ajustement des limites de l'axe x
        ax.set_xlim(left=epoch_start)
        ax2.set_xlim(left=epoch_start)
        ax3.set_xlim(left=epoch_start)
    
        last_train = str(round(float(max(self.all_train_correct)), 2))
        last_val = str(round(float(max(self.all_val_correct)), 2))
        last_test = str(round(self.test_correct, 2))
    
        train_area = simpson(self.all_train_correct, dx=1)
        val_area = simpson(self.all_val_correct, dx=1)
    
        overfitting_area = train_area - val_area
        overfitting_percentage = str(round(100 * overfitting_area / val_area, 2))
    
        annotation_text = ("#:"+ f"{self.model_name}\t"
                           + f"Train: {last_train}%\t"
                           + f"Validation: {last_val}%\t"
                           + f"Test: {last_test}%\t"
                           + f"Overfitting: {overfitting_percentage}%\t\t"
                           + f"Model size: {self.nb_neurones}").expandtabs()
    
        # Ajustement de la position de l'annotation
        ax.annotate(annotation_text, xy=(0.02, 1.05), xycoords='axes fraction', va='bottom')
    
        plt.tight_layout()    
        plt.savefig(self.model_name + '.png')
        plt.show()

    def show_errors(self, dataset, max_errors: int = 9):
        """Affiche les erreurs de prédiction du modèle sur un dataset donné (max 9 erreurs)."""
        images = dataset['image']
        labels = dataset['label']
        
        num_errors = 0
        for i in range(len(images)):
            image = images[i][None, ...]  # Ajouter une dimension batch
            label = int(labels[i])
    
            # Appliquer le modèle
            variables = {'params': self.state.params, 'batch_stats': self.batch_stats}
            logits = self.model.apply(variables, image, mutable=False, train=False)
            pred = int(jnp.argmax(logits, axis=-1)[0])  # Prendre la prédiction
    
            if pred != label:
                num_errors += 1
                img_np = np.transpose(np.array(images[i]), (1, 2, 0))  # [C, H, W] -> [H, W, C]
                
                plt.subplot(3, 3, num_errors)
                plt.imshow(img_np)
                plt.title(f"Vrai: {label}, Prédit: {pred}")
                plt.axis('off')
                
                if num_errors >= max_errors:
                    break
    
        if num_errors > 0:
            plt.suptitle("Exemples d'erreurs de prédiction")
            plt.tight_layout()
            plt.show()
        else:
            print("Aucune erreur détectée sur le dataset.")

    def show_errors_from_pth(self, dataset, mean, std, pth_path, json_path, err_png_path, max_errors: int = 9, use_subset: bool = True):
        """
        Affiche les erreurs de prédiction du modèle chargé depuis un .pth et un .json sur un dataset donné (max 9 erreurs).
        Version optimisée pour CPU : conversion différée et sous-ensemble pour train.
        """
        # Charger la config JSON
        with open(json_path, 'r') as f:
            config = json.load(f)
        model, config = build_model_from_json(config)

        # Charger les poids et batch_stats
        with open(pth_path, 'rb') as f:
            model_data = pickle.load(f)
        params = model_data['params']
        batch_stats = model_data.get('batch_stats', {})

        dummy_rng = jax.random.PRNGKey(0)

        images = dataset['image']
        labels = np.array(dataset['label'])
        
        # ⚠️ Sous-ensemble intelligent : plus d'images si peu d'erreurs attendues
        if use_subset and len(images) > 1000:  # Si dataset large, prendre un sous-ensemble
            num_total = len(images)
            # Pour les erreurs : plus d'images pour capturer toutes les erreurs
            num_subset = min(2000, num_total // 5)  # Max 2000 images pour plus d'erreurs
            np.random.seed(42)  # Fixe pour reproductibilité
            subset_indices = np.random.choice(num_total, num_subset, replace=False)
            images = images[subset_indices]
            labels = labels[subset_indices]
        
        batch_size = 32  # Batch size réduit pour économiser la mémoire
        variables = {'params': params, 'batch_stats': batch_stats}
        all_logits = []
        
        # Prédiction batchée avec conversion différée
        for i in tqdm.tqdm(range(0, len(images), batch_size), desc='Prédiction batchée'):
            batch_images = jnp.array(images[i:i+batch_size])  # Conversion différée ici
            logits = model.apply(variables, batch_images, config, rng=dummy_rng, deterministic=True, mutable=False)
            all_logits.append(np.array(logits))
        all_logits = np.concatenate(all_logits, axis=0)
        preds = np.argmax(all_logits, axis=-1)
        # Trouver les indices des erreurs
        error_indices = np.where(preds != labels)[0]

        plt.figure(figsize=(8, 8))
        for num_errors, idx in enumerate(error_indices[:max_errors], 1):
            img = np.array(images[idx])
            if img.ndim == 3 and img.shape[-1] == 1:
                img_np = img.squeeze(-1)
            elif img.ndim == 3 and img.shape[0] == 1:
                img_np = img.squeeze(0)
            else:
                img_np = img
            
            # Dénormalisation si RGB et mean/std fournis
            if (
                img_np.ndim == 3 and img_np.shape[-1] == 3
                and mean is not None and std is not None
                and len(mean) > 0 and len(std) > 0
            ):
                mean = np.array(mean)
                std = np.array(std)
                img_np = (img_np * std) + mean
                img_np = np.clip(img_np, 0, 1)
            plt.subplot(3, 3, num_errors)
            plt.imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
            # Utilisation des noms de classes si disponibles
            label = int(labels[idx])
            pred = int(preds[idx])
            true_label = self.label_names[label] if self.label_names and label < len(self.label_names) else str(label)
            pred_label = self.label_names[pred] if self.label_names and pred < len(self.label_names) else str(pred)
            ax = plt.gca()
            for t in ax.texts:
                t.remove()
            ax.text(0.5, 1.05, true_label, color='green', fontsize=14, ha='right', va='bottom', transform=ax.transAxes)
            ax.text(0.5, 1.05, ' ' + pred_label, color='red', fontsize=14, ha='left', va='bottom', transform=ax.transAxes)
            plt.axis('off')
        if len(error_indices) > 0:
            plt.tight_layout()
            plt.savefig(err_png_path)
            plt.show()
        else:
            print("Aucune erreur détectée sur le dataset.")

    def confusion_matrix_from_pth(self, dataset, pth_path, json_path, confusion_matrix_png_path, use_subset: bool = True):
        """
        Crée une matrice de confusion du modèle chargé depuis un .pth et un .json sur un dataset donné.
        Version optimisée pour CPU : conversion différée et sous-ensemble pour train.
        """
        # Charger la config JSON
        with open(json_path, 'r') as f:
            config = json.load(f)
        model, config = build_model_from_json(config)

        # Charger les poids et batch_stats
        with open(pth_path, 'rb') as f:
            model_data = pickle.load(f)
        params = model_data['params']
        batch_stats = model_data.get('batch_stats', {})

        dummy_rng = jax.random.PRNGKey(0)

        images = dataset['image']
        labels = np.array(dataset['label'])
        
        # ⚠️ Sous-ensemble intelligent : plus d'images pour matrice de confusion
        if use_subset and len(images) > 1000:  # Si dataset large, prendre un sous-ensemble
            num_total = len(images)
            # Pour la matrice de confusion : plus d'images pour statistiques fiables
            num_subset = min(3000, num_total // 3)  # Max 3000 images pour statistiques
            np.random.seed(42)  # Fixe pour reproductibilité
            subset_indices = np.random.choice(num_total, num_subset, replace=False)
            images = images[subset_indices]
            labels = labels[subset_indices]
        
        # Déterminer le nombre de classes dynamiquement
        num_classes = len(np.unique(labels))
        
        batch_size = 32  # Batch size réduit pour économiser la mémoire
        variables = {'params': params, 'batch_stats': batch_stats}
        all_logits = []
        
        # Prédiction batchée avec conversion différée
        for i in tqdm.tqdm(range(0, len(images), batch_size), desc='Prédiction pour matrice de confusion'):
            batch_images = jnp.array(images[i:i+batch_size])  # Conversion différée ici
            logits = model.apply(variables, batch_images, config, rng=dummy_rng, deterministic=True, mutable=False)
            all_logits.append(np.array(logits))
        
        all_logits = np.concatenate(all_logits, axis=0)
        preds = np.argmax(all_logits, axis=-1)
        
        # Créer la matrice de confusion
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true_label, pred_label in zip(labels, preds):
            confusion_matrix[true_label, pred_label] += 1
        
        # Créer la figure
        plt.figure(figsize=(10, 8))
        
        # Utiliser les noms de classes si disponibles, sinon utiliser les indices
        if self.label_names and len(self.label_names) >= num_classes:
            class_names = self.label_names[:num_classes]
        else:
            class_names = [f"Classe {i}" for i in range(num_classes)]
        
        # Afficher la matrice de confusion
        im = plt.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
        plt.colorbar(im)
        
        # Ajouter les annotations dans chaque cellule
        for i in range(num_classes):
            for j in range(num_classes):
                value = confusion_matrix[i, j]
                color = 'white' if value > confusion_matrix.max() / 2 else 'black'
                plt.text(j, i, str(value), ha='center', va='center', color=color, fontsize=10)
        
        # Configurer les axes
        plt.xticks(range(num_classes), class_names, rotation=45, ha='right')
        plt.yticks(range(num_classes), class_names)
        plt.xlabel('Prédictions')
        plt.ylabel('Vraies classes')
        plt.title('Matrice de Confusion')
        
        # Calculer et afficher les métriques
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        
        # Éviter la division par zéro
        precision = np.nan_to_num(precision, nan=0.0)
        recall = np.nan_to_num(recall, nan=0.0)
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_score = np.nan_to_num(f1_score, nan=0.0)
        
        # Afficher les métriques globales
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1_score)
        
        metrics_text = f'Accuracy: {accuracy:.3f}\nMacro Precision: {macro_precision:.3f}\nMacro Recall: {macro_recall:.3f}\nMacro F1: {macro_f1:.3f}'
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig(confusion_matrix_png_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Matrice de confusion sauvegardée dans {confusion_matrix_png_path}")
        print(f"Accuracy globale: {accuracy:.3f}")
        print(f"Macro Precision: {macro_precision:.3f}")
        print(f"Macro Recall: {macro_recall:.3f}")
        print(f"Macro F1: {macro_f1:.3f}")
        
        return confusion_matrix, accuracy, macro_precision, macro_recall, macro_f1


