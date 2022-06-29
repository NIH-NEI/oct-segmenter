import json
import numpy as np
import PIL.Image
import PIL.ImageDraw

mask = np.zeros((496, 992), dtype=np.uint8)
mask = PIL.Image.fromarray(mask)
draw = PIL.ImageDraw.Draw(mask)

with open("points.json", "r") as f:
    points = json.load(f)

xy = [tuple(point) for point in points]
draw.polygon(xy=xy, outline=1, fill=1)
mask = np.array(mask, dtype=bool)
np.savetxt("pil-test/maskfint_pilv9_1_1.txt", mask, fmt="%d", delimiter="")