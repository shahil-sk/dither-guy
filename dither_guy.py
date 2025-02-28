import tkinter as tk
from tkinter import filedialog, colorchooser, ttk
import sv_ttk
from PIL import Image, ImageTk
import numpy as np
import random

def apply_dither(img, pixel_size, threshold, replace_color, method):
    img = img.convert('L')
    img = img.resize((img.width // pixel_size, img.height // pixel_size), Image.NEAREST)
    img_array = np.array(img, dtype=np.uint8)

    if method == "Bayer":
        bayer_matrix = np.array([
            [  0, 128,  32, 160],
            [192,  64, 224,  96],
            [ 48, 176,  16, 144],
            [240, 112, 208,  80]
        ])
        bayer_matrix = (bayer_matrix / 255.0) * threshold
        for y in range(img_array.shape[0]):
            for x in range(img_array.shape[1]):
                img_array[y, x] = 255 if img_array[y, x] > bayer_matrix[y % 4, x % 4] else 0

    elif method == "Floyd-Steinberg":
        for y in range(img_array.shape[0] - 1):
            for x in range(1, img_array.shape[1] - 1):
                old_pixel = img_array[y, x]
                new_pixel = 255 if old_pixel > threshold else 0
                img_array[y, x] = new_pixel
                quant_error = old_pixel - new_pixel
                img_array[y, x + 1] += quant_error * 7 / 16
                img_array[y + 1, x - 1] += quant_error * 3 / 16
                img_array[y + 1, x] += quant_error * 5 / 16
                img_array[y + 1, x + 1] += quant_error * 1 / 16

    elif method == "Random":
        for y in range(img_array.shape[0]):
            for x in range(img_array.shape[1]):
                img_array[y, x] = 255 if img_array[y, x] > random.randint(0, 255) else 0

    if method == "Atkinson":
        for y in range(img_array.shape[0] - 2):
            for x in range(1, img_array.shape[1] - 2):
                old_pixel = img_array[y, x]
                new_pixel = 255 if old_pixel > threshold else 0
                img_array[y, x] = new_pixel
                quant_error = (old_pixel - new_pixel) // 8
                for dy, dx in [(0, 1), (0, 2), (1, -1), (1, 0), (1, 1), (2, 0)]:
                    if 0 <= y + dy < img_array.shape[0] and 0 <= x + dx < img_array.shape[1]:
                        img_array[y + dy, x + dx] = np.clip(img_array[y + dy, x + dx] + quant_error, 0, 255)

    elif method == "Jarvis-Judice-Ninke":
        error_distribution = np.array([
            [0, 0, 0, 7, 5],
            [3, 5, 7, 5, 3],
            [1, 3, 5, 3, 1]
        ])
        for y in range(img_array.shape[0] - 2):
            for x in range(2, img_array.shape[1] - 2):
                old_pixel = img_array[y, x]
                new_pixel = 255 if old_pixel > threshold else 0
                img_array[y, x] = new_pixel
                quant_error = (old_pixel - new_pixel) / 48
                for dy in range(3):
                    for dx in range(5):
                        ny, nx = y + dy, x + dx - 2
                        if 0 <= ny < img_array.shape[0] and 0 <= nx < img_array.shape[1]:
                            img_array[ny, nx] = np.clip(img_array[ny, nx] + quant_error * error_distribution[dy][dx], 0, 255)

    elif method == "Stucki":
        error_distribution = np.array([
            [0, 0, 0, 8, 4],
            [2, 4, 8, 4, 2],
            [1, 2, 4, 2, 1]
        ])
        for y in range(img_array.shape[0] - 2):
            for x in range(2, img_array.shape[1] - 2):
                old_pixel = img_array[y, x]
                new_pixel = 255 if old_pixel > threshold else 0
                img_array[y, x] = new_pixel
                quant_error = (old_pixel - new_pixel) / 42
                for dy in range(3):
                    for dx in range(5):
                        ny, nx = y + dy, x + dx - 2
                        if 0 <= ny < img_array.shape[0] and 0 <= nx < img_array.shape[1]:
                            img_array[ny, nx] = np.clip(img_array[ny, nx] + quant_error * error_distribution[dy][dx], 0, 255)

    img_array = np.clip(img_array, 0, 255).astype(np.uint8)  # Converte de volta para uint8 após os cálculos
    img = Image.fromarray(img_array)
    img = img.resize((img.width * pixel_size, img.height * pixel_size), Image.NEAREST)
    img = img.convert("RGB")
    img_data = np.array(img)
    img_data[(img_data[:, :, 0] == 255) & (img_data[:, :, 1] == 255) & (img_data[:, :, 2] == 255)] = replace_color
    return Image.fromarray(img_data)

def update_image():
    if original_img:
        dithered_img = apply_dither(original_img, pixel_size.get(), threshold.get(), replace_color.get(), dithering_method.get())
        img_tk = ImageTk.PhotoImage(dithered_img)
        img_label.config(image=img_tk)
        img_label.image = img_tk

def open_file():
    global original_img
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.*")])
    if file_path:
        original_img = Image.open(file_path)
        update_image()

def save_file():
    if original_img:
        dithered_img = apply_dither(original_img, pixel_size.get(), threshold.get(), replace_color.get(), dithering_method.get())
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
        if save_path:
            dithered_img.save(save_path)

def choose_color():
    color = colorchooser.askcolor(color="#658a00")[1]
    if color:
        replace_color.set(tuple(int(color[i:i+2], 16) for i in (1, 3, 5)))

def invert_colors():
    global original_img
    if original_img:
        img_data = np.array(original_img.convert("RGB"))
        img_data = 255 - img_data
        inverted_img = Image.fromarray(img_data)
        original_img = inverted_img
        update_image()

root = tk.Tk()
root.title("Dither Guy")
root.geometry("900x700")

original_img = None
frame_main = tk.Frame(root)
frame_main.pack(pady=5, fill='both', expand=True)

frame_image = tk.Frame(frame_main)
frame_image.pack(side='left', fill='both', expand=True)

frame_buttons = tk.Frame(frame_main)
frame_buttons.pack(side='right', fill='y')

dithering_method = tk.StringVar(value="Bayer")
dropdown = ttk.OptionMenu(frame_buttons, dithering_method, "Bayer", "Floyd-Steinberg", "Bayer","Random", "Atkinson", "Jarvis-Judice-Ninke", "Stucki", command=lambda x: update_image())
dropdown.pack(padx=5, pady=5)

btn_open = ttk.Button(frame_buttons, text="Open Image", command=open_file)
btn_open.pack(padx=5, pady=5)

btn_save = ttk.Button(frame_buttons, text="Save Image", command=save_file)
btn_save.pack(padx=5, pady=5)

btn_color = ttk.Button(frame_buttons, text="Change Color", command=choose_color)
btn_color.pack(padx=5, pady=5)

btn_invert = ttk.Button(frame_buttons, text="Invert", command=invert_colors)
btn_invert.pack(padx=5, pady=5)

pixel_size = tk.IntVar(value=4)
threshold = tk.IntVar(value=128)
replace_color = tk.Variable(value=(255, 255, 255))

tk.Label(frame_buttons, text="Pixel Size:").pack(padx=5, pady=5)
ttk.Scale(frame_buttons, from_=1, to=20, orient='horizontal', variable=pixel_size, command=lambda x: update_image()).pack(padx=5, pady=5)

tk.Label(frame_buttons, text="Threshold:").pack(padx=5, pady=5)
ttk.Scale(frame_buttons, from_=0, to=400, orient='horizontal', variable=threshold, command=lambda x: update_image()).pack(padx=5, pady=5)

img_label = tk.Label(frame_image)
img_label.pack(fill='both', expand=True)

sv_ttk.set_theme("dark")

root.mainloop()
