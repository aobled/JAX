import os
from tkinter import Tk, Canvas, PhotoImage, Label
from PIL import Image, ImageTk, ImageDraw
import uuid

class PhotoViewer:
    def __init__(self, root_folder):
        self.root = Tk()
        self.root.title("Photo Viewer")
        self.root.geometry("1920x1080")
        self.cusor_size = 256

        self.root_folder = root_folder
        self.crops_folder = 'resized/'
        self.save_folder = 'save/'
        self.image_list = sorted([f for f in os.listdir(self.root_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))])
        self.current_image_index = 0

        self.canvas = Canvas(self.root, cursor="cross")
        self.canvas.pack(fill="both", expand=True)
        
        self.zoom_factor = 1
        self.zoom_step = 1.01
        self.drag_step = 1
        self.load_image()

        self.root.bind("<KeyRelease-n>", self.show_next_image)
        self.root.bind("<Button-4>", self.zoom_in)
        self.root.bind("<Button-5>", self.zoom_out)
        self.root.bind("<B1-Motion>", self.display_cursor_square)
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.root.bind("<B3-Motion>", self.drag_move)
        # Lier la fonction pour quitter l'application à la touche "Escape"
        self.root.bind("<Escape>", lambda event: self.quit_app())
        # Récupérer les touches tapés
        self.root.bind("<KeyRelease>", lambda event: self.handle_key_press(event))

        self.draw_cursor_square()

        self.root.mainloop()

    def handle_key_press(self, event):
        # Vérifiez si la touche fait partie du pavé numérique
        if event.keysym.startswith('KP_'):
            numeric_key = event.keysym  
            match numeric_key:
                case 'KP_Insert':
                    self.zoom_factor = 1
                    self.zoom()
                case 'KP_End':
                    self.zoom_factor = 0.1
                    self.zoom()
                case 'KP_Down':
                    self.zoom_factor = 0.2
                    self.zoom()
                case 'KP_Next':
                    self.zoom_factor = 0.3
                    self.zoom()
                case 'KP_Left':
                    self.zoom_factor = 0.4
                    self.zoom()
                case 'KP_Begin':
                    self.zoom_factor = 0.5
                    self.zoom()
                case 'KP_Right':
                    self.zoom_factor = 0.6
                    self.zoom()
                case 'KP_Home':
                    self.zoom_factor = 0.7
                    self.zoom()
                case 'KP_Up':
                    self.zoom_factor = 0.8
                    self.zoom()
                case 'KP_Prior':
                    self.zoom_factor = 0.9
                    self.zoom()
        else:
            if event.keysym == 'c':
                self.save_cropped_image()
            if event.keysym == 'Delete':
                os.remove(self.image_path)
                self.show_next_image(event)
            if event.keysym == 'Right':
                self.show_next_image(event)
            if event.keysym == 'Left':
                self.show_next_image(event, reverse=1)
            if event.keysym == 's':                
                self.image_save_folder()
                self.show_next_image(event)
         
        
    def quit_app(self):
        self.root.destroy()
        
    def load_image(self):
        self.image_path = os.path.join(self.root_folder, self.image_list[self.current_image_index])
        self.original_image = Image.open(self.image_path)
        self.image = self.original_image.copy()
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.config(width=self.image.width, height=self.image.height)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.root.title(self.image_path+" (Zoom: "+str(round(self.zoom_factor,3))+')')

    def show_next_image(self, event, reverse=0):
        if (reverse):
            self.current_image_index = (self.current_image_index - 1) % len(self.image_list)
        else:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_list)
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.zoom_factor = 1
        self.load_image()
        self.draw_cursor_square()

    def zoom_in(self, event):
        self.zoom_factor *= self.zoom_step
        self.zoom()

    def zoom_out(self, event):
        self.zoom_factor /= self.zoom_step
        self.zoom()
        
    def zoom(self):
        # test if image is wider or heigher
        little_rang = 0.98
        if (self.original_image.height > self.original_image.width):
            zoom_factor_max = self.cusor_size/(self.original_image.width*little_rang)
        else:
            zoom_factor_max = self.cusor_size/(self.original_image.height*little_rang)
        
        # Apply zoom_factor_max
        if (self.zoom_factor < zoom_factor_max):
            self.zoom_factor = zoom_factor_max
        new_width = int(self.original_image.width * self.zoom_factor)
        new_height = int(self.original_image.height * self.zoom_factor)

        self.image = self.original_image.resize((new_width, new_height))
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.root.title(self.image_path+" (Zoom: "+str(round(self.zoom_factor,3))+')')

    def drag_move(self, event):
        x, y = event.x, event.y
        delta_x = x - int(self.image.width/2) + self.drag_step
        delta_y = y - int(self.image.height/2) + self.drag_step
        
        self.canvas.scan_dragto(-delta_x, -delta_y, gain=1)
        self.drag_start_x, self.drag_start_y = x, y
        
    def display_cursor_square(self, event):
        x, y = event.x, event.y
        self.draw_cursor_square(x, y)
        #self.drag_start_x, self.drag_start_y = x, y

    def draw_cursor_square(self, x=None, y=None):
        if x is None or y is None:
            x, y = self.root.winfo_pointerxy()
            x -= self.root.winfo_rootx()
            y -= self.root.winfo_rooty()

        x1, y1 = x - self.cusor_size/2, y - self.cusor_size/2
        x2, y2 = x + self.cusor_size/2, y + self.cusor_size/2

        self.canvas.delete("cursor_square")
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="cursor_square")

    def save_cropped_image(self):
        x1, y1, x2, y2 = self.canvas.coords("cursor_square")
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        if x1 < x2 and y1 < y2:
            cropped_region = self.image.crop((x1, y1, x2, y2))
            #file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            unique_filename = str(uuid.uuid4()) + '.png'
            file_path = self.root_folder+'/'+self.crops_folder+unique_filename

            if file_path:
                cropped_region.save(file_path)
        else:
            print("Invalid region selected.")
        
        self.root.title(self.image_path+" (Crop to: "+file_path+')')

    def image_save_folder(self):
        origin = self.image_path
        target = self.root_folder+'/'+self.save_folder+os.path.basename(self.image_path)
        print('save',origin,'to',target)
        os.rename(origin, target)

if __name__ == "__main__":
    root_folder = '/home/aobled/Downloads/tmp'
    viewer = PhotoViewer(root_folder)
