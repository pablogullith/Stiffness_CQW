import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Re-extract page 6 at high resolution 300 dpi
import subprocess
subprocess.run(['pdftoppm', '-png', '-r', '300', '-f', '6', '-l', '6', '____Stiffness_in_Confined_Quantum_Walks (8).pdf', 'page6'])

img = Image.open('page6-06.png')
w, h = img.size
fig2 = img.crop((int(0.05*w), int(0.06*h), int(0.95*w), int(0.53*h)))
cw, ch = fig2.size

# Let's crop each panel (a) to (f)
# Row 1: y in [0.03*ch, 0.28*ch]
# Row 2: y in [0.28*ch, 0.52*ch]
# Col 1: x in [0.03*cw, 0.35*cw]
# Col 2: x in [0.35*cw, 0.67*cw]
# Col 3: x in [0.67*cw, 0.99*cw]

panels = {
    'a': fig2.crop((int(0.03*cw), int(0.03*ch), int(0.35*cw), int(0.28*ch))),
    'b': fig2.crop((int(0.35*cw), int(0.03*ch), int(0.67*cw), int(0.28*ch))),
    'c': fig2.crop((int(0.67*cw), int(0.03*ch), int(0.99*cw), int(0.28*ch))),
    'd': fig2.crop((int(0.03*cw), int(0.28*ch), int(0.35*cw), int(0.52*ch))),
    'e': fig2.crop((int(0.35*cw), int(0.28*ch), int(0.67*cw), int(0.52*ch))),
    'f': fig2.crop((int(0.67*cw), int(0.28*ch), int(0.99*cw), int(0.52*ch))),
}

for k, p in panels.items():
    p.save(f'orig_panel_{k}.png')

print("All original panels a-f saved!")

