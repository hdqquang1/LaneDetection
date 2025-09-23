# -*- coding: utf-8 -*-
import os, glob
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Carpeta con imágenes y labels
folder = "/home/its/Project/dataset/driver_00_30frame"

# Parámetro: salto (avanza de `step` en `step`)
step = 10

# Listar imágenes
img_files = sorted(glob.glob(os.path.join(folder, "*.jpg")))

def parse_lines(txt_file, w, h):
    if not os.path.exists(txt_file):
        return []
    with open(txt_file, "r") as f:
        tokens = f.read().strip().split()

    vals = []
    for t in tokens:
        try:
            vals.append(float(t))
        except:
            continue

    if len(vals) % 2 != 0:
        vals = vals[:-1]

    coords = [(vals[i], vals[i+1]) for i in range(0, len(vals), 2)]
    # Detectar si están como (y, x)
    if coords:
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        # Si algún x está fuera del rango usual, intercambiar
        if max(xs) > w or min(xs) < 0:
            coords = [(c[1], c[0]) for c in coords]

    # Clampear a los límites de la imagen
    coords = [(max(0, min(w-1, x)), max(0, min(h-1, y))) for (x,y) in coords]
    return coords

i = 0
n = len(img_files)
if n == 0:
    raise SystemExit("No se encontraron imágenes en la carpeta especificada.")

while i < n:
    img_path = img_files[i]
    base = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(folder, base + ".lines.txt")  # ajuste según formato de tus txt

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)

    coords = parse_lines(txt_path, w, h)
    if len(coords) > 1:
        draw.line(coords, width=3, fill=(255,0,0))
    for (x,y) in coords:
        r = 4
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(0,255,0))

    plt.figure(figsize=(12,6))
    plt.imshow(overlay)
    plt.title(f"{i+1}/{n}  {os.path.basename(img_path)}")
    plt.axis("off")
    plt.show()

    # Avanza `step`. Comandos simples: Enter = siguiente salto, b = retroceder un salto, q = salir
    cmd = input("Enter=next, b=back, q=quit: ").strip().lower()
    if cmd == "q":
        break
    if cmd == "b":
        i = max(0, i - step)
    else:
        i += step
