import tkinter as tk
from tkinter import filedialog, colorchooser
from PIL import Image, ImageTk
import numpy as np
import cv2

def apply_dither(img, pixel_size, threshold, replace_color):
    img = img.convert('L')
    img = img.resize((img.width // pixel_size, img.height // pixel_size), Image.NEAREST)
    img_array = np.array(img, dtype=np.uint8)

    bayer_matrix = np.array([
        [  0, 128,  32, 160],
        [192,  64, 224,  96],
        [ 48, 176,  16, 144],
        [240, 112, 208,  80]
    ])

    bayer_matrix = (bayer_matrix / 255.0) * threshold

    for y in range(img_array.shape[0]):
        for x in range(img_array.shape[1]):
            if img_array[y, x] > bayer_matrix[y % 4, x % 4]:
                img_array[y, x] = 255
            else:
                img_array[y, x] = 0

    img = Image.fromarray(img_array)
    img = img.resize((img.width * pixel_size, img.height * pixel_size), Image.NEAREST)

    img = img.convert("RGB")
    img_data = np.array(img)

    img_data[(img_data[:, :, 0] == 255) & (img_data[:, :, 1] == 255) & (img_data[:, :, 2] == 255)] = replace_color

    img = Image.fromarray(img_data)
    return img

def update_image():
    if original_img:
        dithered_img = apply_dither(original_img, pixel_size.get(), threshold.get(), replace_color.get())
        img_tk = ImageTk.PhotoImage(dithered_img)
        img_label.config(image=img_tk)
        img_label.image = img_tk

def open_file():
    global original_img, video_cap, is_video
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    if file_path:
        if file_path.endswith(('.mp4', '.avi')):
            video_cap = cv2.VideoCapture(file_path)
            is_video = True
            show_frame() 
        else:  
            original_img = Image.open(file_path)
            is_video = False
            update_image()

def start_saving():
    global video_writer
    if is_video and video_cap.isOpened():
        save_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 Files", "*.mp4")])
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
            fps = video_cap.get(cv2.CAP_PROP_FPS)
            width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
            height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
            video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

            process_video_for_saving()


def process_video_for_saving():
    global video_writer
    if video_cap.isOpened():
        ret, frame = video_cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            dithered_img = apply_dither(img, pixel_size.get(), threshold.get(), replace_color.get())
            
            dithered_img_bgr = np.array(dithered_img)
            dithered_img_bgr = cv2.cvtColor(dithered_img_bgr, cv2.COLOR_RGB2BGR)

            video_writer.write(dithered_img_bgr)
            
            img_tk = ImageTk.PhotoImage(dithered_img)
            img_label.config(image=img_tk)
            img_label.image = img_tk

            img_label.after(int(1000 / video_cap.get(cv2.CAP_PROP_FPS)), process_video_for_saving)
        else:
            video_writer.release()
            video_writer = None
    else:
        print("Erro ao abrir o vídeo.")



def show_frame():
    global video_writer
    if is_video and video_cap.isOpened():
        ret, frame = video_cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            dithered_img = apply_dither(img, pixel_size.get(), threshold.get(), replace_color.get())
            
            dithered_img_bgr = np.array(dithered_img)
            dithered_img_bgr = cv2.cvtColor(dithered_img_bgr, cv2.COLOR_RGB2BGR)

            if video_writer is not None:
                video_writer.write(dithered_img_bgr)
            
            img_tk = ImageTk.PhotoImage(dithered_img)
            img_label.config(image=img_tk)
            img_label.image = img_tk

            img_label.after(int(1000 / video_cap.get(cv2.CAP_PROP_FPS)), show_frame)
        else:
            print("Erro ao ler o frame do vídeo. Verifique se o arquivo de vídeo está válido.")
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  
            show_frame() 
    else:
        print("Erro: O vídeo não pôde ser aberto ou não há mais frames para ler.")


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
video_cap = None
video_writer = None
is_video = False

frame_buttons = tk.Frame(root)
frame_buttons.pack(pady=5)

btn_open = tk.Button(frame_buttons, text="Abrir Video", command=open_file)
btn_open.grid(row=0, column=0, padx=5, pady=5)

btn_save_video = tk.Button(frame_buttons, text="Salvar Video", command=start_saving)
btn_save_video.grid(row=0, column=1, padx=5, pady=5)

btn_color = tk.Button(frame_buttons, text="Escolher Cor para Substituir", command=choose_color)
btn_color.grid(padx=5, pady=5)


pixel_size = tk.IntVar(value=4)
threshold = tk.IntVar(value=128)
replace_color = tk.Variable(value=(255, 255, 255)) 

frame = tk.Frame(root)
frame.pack()

tk.Label(frame, text="Pixelaridade:").pack(side='left')
tk.Scale(frame, from_=1, to=20, orient='horizontal', variable=pixel_size, command=lambda x: update_image()).pack(side='left')

tk.Label(frame, text="Threshold:").pack(side='left')
tk.Scale(frame, from_=0, to=400, orient='horizontal', variable=threshold, command=lambda x: update_image()).pack(side='left')

img_label = tk.Label(root)
img_label.pack()

root.mainloop()
