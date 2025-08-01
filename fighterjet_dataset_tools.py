import os
import json
import glob
import random
import numpy as np
from PIL import Image, ImageOps
import shutil
import tqdm

"""
Corp des images en 224*224 avec bande vertes vertical ou horizontale et fond vert
"""
# ========== 1. Extraction depuis JSONs ==========
def process_dataset(
    root_dir,
    output_dir,
    target_size=64,
    padding_color=(0, 255, 0)
):
    os.makedirs(output_dir, exist_ok=True)

    # Recherche de tous les .jpg et .png
    jpg_files = glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)
    png_files = glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
    image_paths = jpg_files + png_files

    for image_path in tqdm.tqdm(image_paths):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image = Image.open(image_path).convert("RGB")

        json_pattern = os.path.join(os.path.dirname(image_path), f"{base_name}_*.json")
        json_files = glob.glob(json_pattern)

        for json_file in json_files:
            with open(json_file, "r") as f:
                data = json.load(f)

            bbox = data["annotation"]["bbox"]
            category = data["annotation"]["category_name"]
            x, y, w, h = map(int, bbox)

            if w <= 0 or h <= 0:
                print(f"[⚠️] Bbox invalide ignorée : {json_file}")
                continue

            cropped = image.crop((x, y, x + w, y + h))

            # Zoom si trop petit
            if cropped.width < target_size and cropped.height < target_size:
                scale = target_size / max(cropped.width, cropped.height)
                new_w = int(cropped.width * scale)
                new_h = int(cropped.height * scale)
                cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Resize + padding (letterbox)
            cropped.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            pad_w = (target_size - cropped.width) // 2
            pad_h = (target_size - cropped.height) // 2
            padded = ImageOps.expand(
                cropped,
                border=(pad_w, pad_h,
                        target_size - cropped.width - pad_w,
                        target_size - cropped.height - pad_h),
                fill=padding_color
            )

            class_dir = os.path.join(output_dir, category)
            os.makedirs(class_dir, exist_ok=True)

            out_name = os.path.splitext(os.path.basename(json_file))[0] + ".png"
            out_path = os.path.join(class_dir, out_name)
            padded.save(out_path)
            #print(f"[✓] Sauvé : {out_path}")

# Exemple d’utilisation :
#process_dataset(root_dir="/home/aobled/Downloads/Figtherjet_DATASET", output_dir="/home/aobled/Downloads/_crop_classification")



def process_dataset_stretched(
    root_dir,
    output_dir,
    target_size=64
):
    os.makedirs(output_dir, exist_ok=True)

    jpg_files = glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)
    png_files = glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
    image_paths = jpg_files + png_files

    for image_path in tqdm.tqdm(image_paths):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image = Image.open(image_path).convert("RGB")

        json_pattern = os.path.join(os.path.dirname(image_path), f"{base_name}_*.json")
        json_files = glob.glob(json_pattern)

        for json_file in tqdm.tqdm(json_files):
            with open(json_file, "r") as f:
                data = json.load(f)

            bbox = data["annotation"]["bbox"]
            category = data["annotation"]["category_name"]
            x, y, w, h = map(int, bbox)

            if w <= 0 or h <= 0:
                print(f"[⚠️] Bbox invalide ignorée : {json_file}")
                continue

            cropped = image.crop((x, y, x + w, y + h))

            # Ici on redimensionne en target_size × target_size SANS respecter le ratio (stretched)
            stretched = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

            # Création dossier classe
            class_dir = os.path.join(output_dir, category)
            os.makedirs(class_dir, exist_ok=True)

            # Sauvegarde avec nom JSON
            out_name = os.path.splitext(os.path.basename(json_file))[0] + ".png"
            out_path = os.path.join(class_dir, out_name)
            stretched.save(out_path)

            #print(f"[✓] Sauvé (stretched) : {out_path}")

#process_dataset_stretched(root_dir="/home/aobled/Downloads/Figtherjet_DATASET", output_dir="/home/aobled/Downloads/_crop_classification")




def reflect_pad_np(img: Image.Image, target_size: int) -> Image.Image:
    """Ajoute du padding miroir à une image PIL pour obtenir un carré target_size×target_size"""
    arr = np.array(img)

    h, w, c = arr.shape
    pad_h = target_size - h
    pad_w = target_size - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # padding par réflexion
    padded = np.pad(
        arr,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='reflect'
    )

    return Image.fromarray(padded)

def process_dataset_reflect_numpy(
    root_dir,
    output_dir,
    target_size=64
):
    os.makedirs(output_dir, exist_ok=True)

    image_paths = glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)

    for image_path in tqdm.tqdm(image_paths):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image = Image.open(image_path).convert("RGB")

        json_pattern = os.path.join(os.path.dirname(image_path), f"{base_name}_*.json")
        json_files = glob.glob(json_pattern)

        for json_file in json_files:
            with open(json_file, "r") as f:
                data = json.load(f)

            bbox = data["annotation"]["bbox"]
            category = data["annotation"]["category_name"]
            x, y, w, h = map(int, bbox)

            if w <= 0 or h <= 0:
                print(f"[⚠️] Bbox invalide ignorée : {json_file}")
                continue

            cropped = image.crop((x, y, x + w, y + h))

            # Resize avec ratio préservé
            cropped.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

            # Padding miroir manuel
            padded = reflect_pad_np(cropped, target_size)

            # Création dossier
            class_dir = os.path.join(output_dir, category)
            os.makedirs(class_dir, exist_ok=True)

            out_name = os.path.splitext(os.path.basename(json_file))[0] + ".png"
            out_path = os.path.join(class_dir, out_name)
            padded.save(out_path)

            #print(f"[✓] Sauvé (mirror numpy) : {out_path}")


#process_dataset_reflect_numpy(root_dir="/home/aobled/Downloads/Figtherjet_DATASET", output_dir="/home/aobled/Downloads/_crop_classification")

from PIL import Image
def process_dataset_cropped_square_centered(
    root_dir,
    output_dir,
    target_size=128,
    scale=1.2
):
    os.makedirs(output_dir, exist_ok=True)

    image_paths = glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)

    for image_path in tqdm.tqdm(image_paths):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image = Image.open(image_path).convert("RGB")
        W, H = image.size

        json_pattern = os.path.join(os.path.dirname(image_path), f"{base_name}_*.json")
        json_files = glob.glob(json_pattern)

        for json_file in json_files:
            with open(json_file, "r") as f:
                data = json.load(f)

            bbox = data["annotation"]["bbox"]
            category = data["annotation"]["category_name"]
            x, y, w, h = map(int, bbox)

            if w <= target_size and h <= target_size:
                print(f"[⚠️] Bbox too small (w/h ≤ {target_size}) : {json_file}")
                continue

            """if w <= 0 or h <= 0:
                print(f"[⚠️] Bbox invalide ignorée (w/h ≤ 0) : {json_file}")
                continue"""

            # Calcule carré centré agrandi
            cx = x + w / 2
            cy = y + h / 2
            max_box = max(w, h)
            box_size = int(min(max_box * scale, min(W, H)))  # pas plus grand que l’image

            left = int(cx - box_size / 2)
            top = int(cy - box_size / 2)
            right = left + box_size
            bottom = top + box_size

            # Clip dans l'image
            left = max(0, left)
            top = max(0, top)
            right = min(W, right)
            bottom = min(H, bottom)

            # Vérifie que la box finale est bien valide
            if right <= left or bottom <= top:
                print(f"[⚠️] Box clippée invalide ignorée : {json_file}")
                print(f"    → left={left}, right={right}, top={top}, bottom={bottom}, image_size=({W},{H})")
                continue

            # Crop + Resize
            cropped = image.crop((left, top, right, bottom))
            resized = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

            # Dossier de classe
            class_dir = os.path.join(output_dir, category)
            os.makedirs(class_dir, exist_ok=True)

            out_name = os.path.splitext(os.path.basename(json_file))[0] + ".png"
            out_path = os.path.join(class_dir, out_name)
            resized.save(out_path)

            #print(f"[✓] Sauvé (crop carré centré) : {out_path}")
            
#process_dataset_cropped_square_centered(root_dir="/home/aobled/Downloads/Figtherjet_DATASET", output_dir="/home/aobled/Downloads/_crop_classification")

# ========== 2. Équilibrage + Split train/val ==========
"""
_balanced_dataset_split/
├── train/
│   ├── f16/
│   ├── b2/
│   └── ...
├── val/
│   ├── f16/
│   ├── b2/
│   └── ..."""
def balance_and_split_dataset(
    input_dir,
    output_dir,
    max_per_class=3027,
    val_ratio=0.1,
    seed=42
):
    random.seed(seed)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".png"))]
        selected = random.sample(images, min(max_per_class, len(images)))

        # Split train / val
        val_count = int(len(selected) * val_ratio)
        val_images = selected[:val_count]
        train_images = selected[val_count:]

        for subset, subset_images in zip(["train", "val"], [train_images, val_images]):
            dest_dir = os.path.join(output_dir, subset, class_name)
            os.makedirs(dest_dir, exist_ok=True)

            for img in subset_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(dest_dir, img)
                shutil.copy(src, dst)

            print(f"[✓] {subset}/{class_name} : {len(subset_images)} images")

#balance_and_split_dataset("/home/aobled/Downloads/_crop_classification", "/home/aobled/Downloads/_balanced_dataset_split")


# ========== 3. Création des fichiers NPZ ==========
def create_npz_splits(
    dataset_dir,
    output_prefix,
    image_size=(128, 128)
):
    for split in ['train', 'val']:
        split_dir = os.path.join(dataset_dir, split)
        class_names = sorted(os.listdir(split_dir))
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        images, labels = [], []

        for class_name in tqdm.tqdm(class_names):
            class_path = os.path.join(split_dir, class_name)
            for path in glob.glob(os.path.join(class_path, "*.png")):
                img = Image.open(path).convert("RGB").resize(image_size)
                images.append(np.asarray(img))
                labels.append(class_to_idx[class_name])

        images = np.stack(images).astype(np.uint8)
        labels = np.array(labels).astype(np.uint8)

        out_file = f"{output_prefix}_{split}.npz"
        np.savez_compressed(out_file, image=images, label=labels)
        print(f"[✓] {split} : {len(images)} images sauvegardées dans {out_file}")


# Étape 3 : conversion en fichiers NPZ JAX-friendly
create_npz_splits("/home/aobled/Downloads/_balanced_dataset_split", "fighterjet")

"""import numpy as np
train = np.load("fighterjet_train.npz")
print(train['image'].shape, train['label'].shape)

data = np.load("fighterjet_train.npz")
images = data["image"].astype(np.float32) / 255.0
mean = images.mean(axis=(0, 1, 2))
std = images.std(axis=(0, 1, 2))

print("mean:", mean)
print("std:", std)

"""


# Calcul du MEAN et STD
def load_images(image_paths):
    images = []
    for img_path in tqdm.tqdm(image_paths):
        with Image.open(img_path) as img:
            # Redimensionner l'image à 32x32 pixels
            #img = img.resize((32, 32))
            # Convertir l'image en tableau numpy
            img_array = np.array(img)
            images.append(img_array)
    return np.array(images)

def calculate_normalization_stats(root_dir):
    # Recherche de tous les fichiers .jpg et .png
    jpg_files = glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)
    png_files = glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
    image_paths = jpg_files + png_files

    if not image_paths:
        raise ValueError("Aucune image trouvée dans le répertoire spécifié. Veuillez vérifier le chemin et réessayer.")

    # Charger les images
    images = load_images(image_paths)

    # Calculer la moyenne et l'écart-type
    mean = np.mean(images / 255.0, axis=(0, 1, 2))
    std = np.std(images / 255.0, axis=(0, 1, 2))

    print("Moyenne:", mean)
    print("Écart-type:", std)
    return mean, std

mean, std = calculate_normalization_stats('/home/aobled/Downloads/_balanced_dataset_split/')
