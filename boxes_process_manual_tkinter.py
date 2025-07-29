import os
import json
import tkinter as tk
from tkinter import Tk, Canvas, PhotoImage, simpledialog
from PIL import Image, ImageTk

class ImageManager:
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.crops_folder = 'resized'
        self.save_folder = 'save'
        self.image_list = []
        self.current_image_index = 0
        self.zoom_factor = 1.0
        self.zoom_step = 1.01
        self.drag_step = 1
        self.cursor_size = 256

        self.setup_folders()
        self.load_images()

    def setup_folders(self):
        os.makedirs(os.path.join(self.root_folder, self.crops_folder), exist_ok=True)
        os.makedirs(os.path.join(self.root_folder, self.save_folder), exist_ok=True)

    def load_images(self):
        self.image_list = sorted([
            f for f in os.listdir(self.root_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))
        ])

    def get_image_path(self):
        return os.path.join(self.root_folder, self.image_list[self.current_image_index])

    def get_bounding_boxes(self):
        base_name = os.path.splitext(self.image_list[self.current_image_index])[0]
        bbox_files = [f for f in os.listdir(self.root_folder) if f.startswith(base_name) and f.endswith('.json')]

        bounding_boxes = []
        for bbox_file in bbox_files:
            with open(os.path.join(self.root_folder, bbox_file), 'r') as f:
                data = json.load(f)
                # Retourner aussi le nom du fichier JSON comme identifiant unique
                bounding_boxes.append((data['annotation']['bbox'], data['annotation']['category_name'], bbox_file))

        return bounding_boxes

class PhotoViewer:
    def __init__(self, root_folder):
        self.root = Tk()
        self.root.title("Photo Viewer")
        self.root.geometry("1920x1080")

        self.image_manager = ImageManager(root_folder)
        self.canvas = Canvas(self.root, cursor="cross")
        self.canvas.pack(fill="both", expand=True)

        self.drag_start_x = 0
        self.drag_start_y = 0
        self.selected_bbox = None
        self.selected_handle = None
        self.bbox_handles = []
        self.bboxes = []
        self.bbox_coords = []
        self.bbox_files = []  # Liste des noms de fichiers JSON pour chaque box
        self.deleted_bboxes = set()  # Maintenant stocke les noms de fichiers JSON
        self.highlighted_bbox = None
        self.dragging_bbox = None
        self.drag_offset_x = 0
        self.drag_offset_y = 0

        self.tk_image = None
        self.original_image = None
        self.image = None

        self.bind_events()
        self.load_image()
        self.draw_bounding_boxes()

        self.root.mainloop()

    def bind_events(self):
        #self.root.bind("<KeyRelease-n>", self.show_next_image)
        self.root.bind("<Button-4>", self.zoom_in)
        self.root.bind("<Button-5>", self.zoom_out)
        self.root.bind("<B1-Motion>", self.on_drag)
        self.root.bind("<Button-1>", self.on_click)
        self.root.bind("<Motion>", self.on_mouse_move)
        self.root.bind("<Button-3>", self.on_right_click)
        self.root.bind("<B3-Motion>", self.on_right_drag)
        self.root.bind("<ButtonRelease-3>", self.on_right_release)
        self.root.bind("<Escape>", lambda event: self.quit_app())
        self.root.bind("<KeyRelease>", self.handle_key_press)

    def handle_key_press(self, event):
        if event.keysym.startswith('KP_'):
            self.handle_numeric_keypad(event.keysym)
        else:
            self.handle_regular_key(event.keysym)

    def handle_numeric_keypad(self, key):
        zoom_values = {
            'KP_Insert': 1.0,
            'KP_End': 0.1,
            'KP_Down': 0.2,
            'KP_Next': 0.3,
            'KP_Left': 0.4,
            'KP_Begin': 0.5,
            'KP_Right': 0.6,
            'KP_Home': 0.7,
            'KP_Up': 0.8,
            'KP_Prior': 0.9
        }
        if key in zoom_values:
            self.image_manager.zoom_factor = zoom_values[key]
            self.zoom()

    def handle_regular_key(self, key):
        actions = {
            'Delete': lambda: self.delete_and_next(),
            'Right': lambda: self.show_next_image(None),
            'Left': lambda: self.show_next_image(None, reverse=True),
            's': self.image_save_folder,
            'd': self.delete_bbox,
            'r': self.edit_category_name,
            'f': self.fill_box_full_image,  # Ajout de l'action 'f'
            'l': self.fill_box_full_width,  # Ajout de l'action 'l'
            'n': self.add_new_box  # Ajout de l'action 'n'
        }
        if key in actions:
            actions[key]()

    def add_new_box(self):
        # Crée une nouvelle boxe en (0,0,50,50) avec une catégorie par défaut
        base_name = os.path.splitext(self.image_manager.image_list[self.image_manager.current_image_index])[0]
        # Chercher le prochain numéro disponible
        existing = [f for f in os.listdir(self.image_manager.root_folder) if f.startswith(base_name) and f.endswith('.json')]
        nums = []
        for f in existing:
            try:
                num = int(f.split('_')[-1].split('.')[0])
                nums.append(num)
            except Exception:
                pass
        next_num = max(nums) + 1 if nums else 0
        new_json_name = f"{base_name}_{next_num}.json"
        new_json_path = os.path.join(self.image_manager.root_folder, new_json_name)
        # Structure du fichier JSON
        data = {
            "annotation": {
                "bbox": [0, 0, 50, 50],
                "category_name": "unknown"
            }
        }
        with open(new_json_path, 'w') as f:
            json.dump(data, f, indent=2)
        # Redessiner les boxes
        self.draw_bounding_boxes()

    def delete_and_next(self):
        image_path = self.image_manager.get_image_path()
        base_name = os.path.splitext(self.image_manager.image_list[self.image_manager.current_image_index])[0]

        os.remove(image_path)

        bbox_files = [f for f in os.listdir(self.image_manager.root_folder) if f.startswith(base_name) and f.endswith('.json')]
        for bbox_file in bbox_files:
            os.remove(os.path.join(self.image_manager.root_folder, bbox_file))

        self.show_next_image(None)

    def load_image(self):
        image_path = self.image_manager.get_image_path()
        self.original_image = Image.open(image_path)
        self.image = self.original_image.copy()
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.config(width=self.image.width, height=self.image.height)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.root.title(f"{image_path} (Zoom: {round(self.image_manager.zoom_factor, 3)})")

    def show_next_image(self, event, reverse=False):
        if reverse:
            self.image_manager.current_image_index = (
                self.image_manager.current_image_index - 1
            ) % len(self.image_manager.image_list)
        else:
            self.image_manager.current_image_index = (
                self.image_manager.current_image_index + 1
            ) % len(self.image_manager.image_list)

        self.drag_start_x = 0
        self.drag_start_y = 0
        self.image_manager.zoom_factor = 1
        self.deleted_bboxes = set()
        self.load_image()
        self.draw_bounding_boxes()

    def zoom_in(self, event):
        self.image_manager.zoom_factor *= self.image_manager.zoom_step
        self.zoom()

    def zoom_out(self, event):
        self.image_manager.zoom_factor /= self.image_manager.zoom_step
        self.zoom()

    def zoom(self):
        little_rang = 0.98
        if self.original_image.height > self.original_image.width:
            zoom_factor_max = self.image_manager.cursor_size / (self.original_image.width * little_rang)
        else:
            zoom_factor_max = self.image_manager.cursor_size / (self.original_image.height * little_rang)

        if self.image_manager.zoom_factor < zoom_factor_max:
            self.image_manager.zoom_factor = zoom_factor_max

        new_width = int(self.original_image.width * self.image_manager.zoom_factor)
        new_height = int(self.original_image.height * self.image_manager.zoom_factor)
        self.image = self.original_image.resize((new_width, new_height))
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.root.title(f"{self.image_manager.get_image_path()} (Zoom: {round(self.image_manager.zoom_factor, 3)})")
        self.draw_bounding_boxes()

    def on_click(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

        clicked_item = self.canvas.find_closest(event.x, event.y)[0]

        if clicked_item in self.bbox_handles:
            self.selected_handle = clicked_item
            handle_index = self.bbox_handles.index(clicked_item)
            self.selected_bbox = handle_index // 4  # 4 poignées par boxe maintenant
        else:
            self.selected_handle = None
            self.selected_bbox = None

    def on_drag(self, event):
        if self.selected_handle is not None:
            delta_x = event.x - self.drag_start_x
            delta_y = event.y - self.drag_start_y

            handle_index = self.bbox_handles.index(self.selected_handle)
            bbox_index = self.selected_bbox
            x1, y1, x2, y2 = self.bbox_coords[bbox_index]

            # 4 poignées par boxe : 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
            handle_in_box = handle_index % 4
            if handle_in_box == 0:  # Top-left handle
                x1 += delta_x
                y1 += delta_y
            elif handle_in_box == 1:  # Top-right handle
                x2 += delta_x
                y1 += delta_y
            elif handle_in_box == 2:  # Bottom-right handle
                x2 += delta_x
                y2 += delta_y
            else:  # Bottom-left handle (handle_in_box == 3)
                x1 += delta_x
                y2 += delta_y

            self.canvas.coords(self.bboxes[bbox_index], x1, y1, x2, y2)
            self.update_bbox_coords(bbox_index, x1, y1, x2, y2)

            self.drag_start_x = event.x
            self.drag_start_y = event.y

    def update_bbox_coords(self, bbox_index, x1, y1, x2, y2):
        self.bbox_coords[bbox_index] = (x1, y1, x2, y2)
        self.update_handles(bbox_index)

    def update_handles(self, bbox_index):
        handle_size = 10
        x1, y1, x2, y2 = self.bbox_coords[bbox_index]

        # 4 poignées par boxe : top-left, top-right, bottom-right, bottom-left
        self.canvas.coords(self.bbox_handles[4 * bbox_index],  # Top-left
                           x1 - handle_size / 2, y1 - handle_size / 2,
                           x1 + handle_size / 2, y1 + handle_size / 2)

        self.canvas.coords(self.bbox_handles[4 * bbox_index + 1],  # Top-right
                           x2 - handle_size / 2, y1 - handle_size / 2,
                           x2 + handle_size / 2, y1 + handle_size / 2)

        self.canvas.coords(self.bbox_handles[4 * bbox_index + 2],  # Bottom-right
                           x2 - handle_size / 2, y2 - handle_size / 2,
                           x2 + handle_size / 2, y2 + handle_size / 2)

        self.canvas.coords(self.bbox_handles[4 * bbox_index + 3],  # Bottom-left
                           x1 - handle_size / 2, y2 - handle_size / 2,
                           x1 + handle_size / 2, y2 + handle_size / 2)

    def draw_bounding_boxes(self):
        self.canvas.delete("bbox")
        self.bboxes = []
        self.bbox_handles = []
        self.bbox_coords = []
        self.bbox_files = []  # Réinitialiser la liste des fichiers JSON

        for idx, (bbox, category_name, bbox_file) in enumerate(self.image_manager.get_bounding_boxes()):
            color = "red" if category_name.lower() == "unknown" else "green"
            if bbox_file not in self.deleted_bboxes:
                x1, y1, width, height = bbox
                x1_zoomed = x1 * self.image_manager.zoom_factor
                y1_zoomed = y1 * self.image_manager.zoom_factor
                x2_zoomed = (x1 + width) * self.image_manager.zoom_factor
                y2_zoomed = (y1 + height) * self.image_manager.zoom_factor

                bbox_id = self.canvas.create_rectangle(x1_zoomed, y1_zoomed, x2_zoomed, y2_zoomed, outline=color, width=2, tags="bbox")
                self.bboxes.append(bbox_id)
                self.bbox_coords.append((x1_zoomed, y1_zoomed, x2_zoomed, y2_zoomed))
                self.bbox_files.append(bbox_file)

                self.canvas.create_text(x1_zoomed + 10, y1_zoomed - 10, text=category_name, fill=color, font=("Arial", 12), tags="bbox", anchor="nw")

                handle_size = 10
                # 4 poignées par boxe : top-left, top-right, bottom-right, bottom-left
                handle1 = self.canvas.create_rectangle(x1_zoomed - handle_size / 2, y1_zoomed - handle_size / 2,
                                                       x1_zoomed + handle_size / 2, y1_zoomed + handle_size / 2,
                                                       fill="blue", tags="bbox")
                handle2 = self.canvas.create_rectangle(x2_zoomed - handle_size / 2, y1_zoomed - handle_size / 2,
                                                       x2_zoomed + handle_size / 2, y1_zoomed + handle_size / 2,
                                                       fill="blue", tags="bbox")
                handle3 = self.canvas.create_rectangle(x2_zoomed - handle_size / 2, y2_zoomed - handle_size / 2,
                                                       x2_zoomed + handle_size / 2, y2_zoomed + handle_size / 2,
                                                       fill="blue", tags="bbox")
                handle4 = self.canvas.create_rectangle(x1_zoomed - handle_size / 2, y2_zoomed - handle_size / 2,
                                                       x1_zoomed + handle_size / 2, y2_zoomed + handle_size / 2,
                                                       fill="blue", tags="bbox")

                self.bbox_handles.extend([handle1, handle2, handle3, handle4])
            else:
                pass

    def on_mouse_move(self, event):
        x, y = event.x, event.y

        if self.highlighted_bbox:
            self.canvas.delete(self.highlighted_bbox)
            self.highlighted_bbox = None

        for idx, (x1, y1, x2, y2) in enumerate(self.bbox_coords):
            # On récupère la couleur de la box associée
            category_name = None
            if idx < len(self.bbox_files):
                bbox_file = self.bbox_files[idx]
                # Chercher la catégorie associée à ce fichier
                for box in self.image_manager.get_bounding_boxes():
                    if box[2] == bbox_file:
                        category_name = box[1]
                        break
            color = "red" if category_name and category_name.lower() == "unknown" else "green"
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.highlighted_bbox = self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, fill=color, stipple="gray25", width=2)
                break
        self.root.title(f"{self.image_manager.get_image_path()} (x:{x}, y:{y})")

    def on_right_click(self, event):
        x, y = event.x, event.y

        for idx, (x1, y1, x2, y2) in enumerate(self.bbox_coords):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.dragging_bbox = idx
                self.drag_offset_x = x - x1
                self.drag_offset_y = y - y1
                break

    def on_right_drag(self, event):
        if self.dragging_bbox is not None:
            x1, y1, x2, y2 = self.bbox_coords[self.dragging_bbox]
            delta_x = event.x - self.drag_offset_x - x1
            delta_y = event.y - self.drag_offset_y - y1

            self.canvas.move(self.bboxes[self.dragging_bbox], delta_x, delta_y)

            self.bbox_coords[self.dragging_bbox] = (x1 + delta_x, y1 + delta_y, x2 + delta_x, y2 + delta_y)

            self.update_handles(self.dragging_bbox)

    def on_right_release(self, event):
        self.dragging_bbox = None

    def delete_bbox(self):
        x, y = self.root.winfo_pointerxy()
        x -= self.root.winfo_rootx()
        y -= self.root.winfo_rooty()

        for idx, (x1, y1, x2, y2) in enumerate(self.bbox_coords):
            if x1 <= x <= x2 and y1 <= y <= y2:
                bbox_file_to_delete = self.bbox_files[idx]
                self.deleted_bboxes.add(bbox_file_to_delete)
                self.draw_bounding_boxes()
                break

    def edit_category_name(self):
        x, y = self.root.winfo_pointerxy()
        x -= self.root.winfo_rootx()
        y -= self.root.winfo_rooty()

        for idx, (x1, y1, x2, y2) in enumerate(self.bbox_coords):
            if x1 <= x <= x2 and y1 <= y <= y2:
                base_name = os.path.splitext(self.image_manager.image_list[self.image_manager.current_image_index])[0]
                bbox_files = [f for f in os.listdir(self.image_manager.root_folder) if f.startswith(base_name) and f.endswith('.json')]

                if self.bbox_files[idx] in bbox_files:
                    bbox_file = self.bbox_files[idx]
                    bbox_file_path = os.path.join(self.image_manager.root_folder, bbox_file)

                    with open(bbox_file_path, 'r') as f:
                        data = json.load(f)

                    new_category_name = simpledialog.askstring("Edit Category", "Enter new category name:", initialvalue=data['annotation']['category_name'])

                    if new_category_name:
                        data['annotation']['category_name'] = new_category_name

                        with open(bbox_file_path, 'w') as f:
                            json.dump(data, f)

                        self.draw_bounding_boxes()
                break

    def fill_box_full_image(self):
        # Si une seule boxe affichée (non supprimée)
        current_boxes = self.image_manager.get_bounding_boxes()
        boxes = [(i, box) for i, box in enumerate(current_boxes) if box[2] not in self.deleted_bboxes]
        print("OK fill_box_full_image")
        if len(boxes) == 1:
            print("resize")
            idx, (bbox, category_name, bbox_file) = boxes[0]
            # Trouver l'index correct dans self.bbox_coords
            bbox_coords_idx = None
            for i, (_, _, file) in enumerate(current_boxes):
                if file == bbox_file and file not in self.deleted_bboxes:
                    bbox_coords_idx = i
                    break
            
            if bbox_coords_idx is not None:
                # Mettre à jour la box pour qu'elle englobe toute l'image
                width, height = self.image.width, self.image.height
                # Mettre à jour les coordonnées zoomées
                x1_zoomed = 0
                y1_zoomed = 0
                x2_zoomed = width
                y2_zoomed = height
                self.bbox_coords[bbox_coords_idx] = (x1_zoomed, y1_zoomed, x2_zoomed, y2_zoomed)
                print(self.bbox_coords[bbox_coords_idx])
                
                # Sauvegarder les nouvelles coordonnées dans le fichier JSON
                bbox_file_path = os.path.join(self.image_manager.root_folder, bbox_file)
                with open(bbox_file_path, 'r') as f:
                    data = json.load(f)
                
                # Convertir les coordonnées zoomées en coordonnées originales
                x1_original = x1_zoomed / self.image_manager.zoom_factor
                y1_original = y1_zoomed / self.image_manager.zoom_factor
                width_original = (x2_zoomed - x1_zoomed) / self.image_manager.zoom_factor
                height_original = (y2_zoomed - y1_zoomed) / self.image_manager.zoom_factor
                
                data['annotation']['bbox'] = [x1_original, y1_original, width_original, height_original]
                
                with open(bbox_file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                self.draw_bounding_boxes()

    def fill_box_full_width(self):
        # Si une seule boxe affichée (non supprimée)
        current_boxes = self.image_manager.get_bounding_boxes()
        print("AVANT fill_box_full_width")
        boxes = [(i, box) for i, box in enumerate(current_boxes) if box[2] not in self.deleted_bboxes]
        print("OK fill_box_full_width")
        if len(boxes) == 1:
            print("resize width")
            idx, (bbox, category_name, bbox_file) = boxes[0]
            # Trouver l'index correct dans self.bbox_coords
            bbox_coords_idx = None
            for i, (_, _, file) in enumerate(current_boxes):
                if file == bbox_file and file not in self.deleted_bboxes:
                    bbox_coords_idx = i
                    break
            
            if bbox_coords_idx is not None:
                # Mettre à jour la box pour qu'elle prenne toute la largeur en conservant sa position verticale
                width, height = self.image.width, self.image.height
                x1, y1, w, h = bbox
                # Mettre à jour les coordonnées zoomées
                x1_zoomed = 0  # Commence à x=0
                y1_zoomed = y1 * self.image_manager.zoom_factor  # Conserve la position Y
                x2_zoomed = width  # Prend toute la largeur
                y2_zoomed = (y1 + h) * self.image_manager.zoom_factor  # Conserve la hauteur
                self.bbox_coords[bbox_coords_idx] = (x1_zoomed, y1_zoomed, x2_zoomed, y2_zoomed)
                print(self.bbox_coords[bbox_coords_idx])
                
                # Sauvegarder les nouvelles coordonnées dans le fichier JSON
                bbox_file_path = os.path.join(self.image_manager.root_folder, bbox_file)
                with open(bbox_file_path, 'r') as f:
                    data = json.load(f)
                
                # Convertir les coordonnées zoomées en coordonnées originales
                x1_original = x1_zoomed / self.image_manager.zoom_factor
                y1_original = y1_zoomed / self.image_manager.zoom_factor
                width_original = (x2_zoomed - x1_zoomed) / self.image_manager.zoom_factor
                height_original = (y2_zoomed - y1_zoomed) / self.image_manager.zoom_factor
                
                data['annotation']['bbox'] = [x1_original, y1_original, width_original, height_original]
                
                with open(bbox_file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                self.draw_bounding_boxes()

    def image_save_folder(self):
        origin_image_path = self.image_manager.get_image_path()
        base_name = os.path.splitext(self.image_manager.image_list[self.image_manager.current_image_index])[0]

        target_image_path = os.path.join(self.image_manager.root_folder, self.image_manager.save_folder, os.path.basename(origin_image_path))
        os.rename(origin_image_path, target_image_path)

        bbox_files = [f for f in os.listdir(self.image_manager.root_folder) if f.startswith(base_name) and f.endswith('.json')]

        # Créer une correspondance entre les noms de fichiers et leurs coordonnées
        bbox_coords_map = {}
        for idx, bbox_file in enumerate(self.bbox_files):
            if idx < len(self.bbox_coords):
                bbox_coords_map[bbox_file] = self.bbox_coords[idx]

        for bbox_file in bbox_files:
            if bbox_file not in self.deleted_bboxes:
                bbox_file_path = os.path.join(self.image_manager.root_folder, bbox_file)

                with open(bbox_file_path, 'r') as f:
                    data = json.load(f)

                # Utiliser les coordonnées mises à jour si disponibles
                if bbox_file in bbox_coords_map:
                    x1, y1, x2, y2 = bbox_coords_map[bbox_file]
                    x1_original = x1 / self.image_manager.zoom_factor
                    y1_original = y1 / self.image_manager.zoom_factor
                    x2_original = x2 / self.image_manager.zoom_factor
                    y2_original = y2 / self.image_manager.zoom_factor

                    data['annotation']['bbox'] = [x1_original, y1_original, x2_original - x1_original, y2_original - y1_original]

                target_bbox_path = os.path.join(self.image_manager.root_folder, self.image_manager.save_folder, bbox_file)
                with open(target_bbox_path, 'w') as f:
                    json.dump(data, f)

                os.remove(bbox_file_path)
            else:
                bbox_file_path = os.path.join(self.image_manager.root_folder, bbox_file)
                os.remove(bbox_file_path)

        self.show_next_image(None)

    def quit_app(self):
        self.root.destroy()

if __name__ == "__main__":
    root_folder = '/home/aobled/Downloads/tmp'
    viewer = PhotoViewer(root_folder)
