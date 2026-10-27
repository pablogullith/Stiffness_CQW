import numpy as np
from PIL import Image

img = Image.open('panel_g_zoom.png')
arr = np.array(img)

# Let's find blue pixels in panel_g_zoom:
# Blue color: high B, low R, low G
mask_blue = (arr[:, :, 2] > 150) & (arr[:, :, 0] < 80) & (arr[:, :, 1] < 80)
# Find y coordinates of blue pixels across x
y_coords = []
x_coords = []
for x in range(arr.shape[1]):
    blues = np.where(mask_blue[:, x])[0]
    if len(blues) > 0:
        y_coords.append(np.mean(blues))
        x_coords.append(x)

print(f"Found {len(x_coords)} blue points in panel_g")
# Let's save a plot of the extracted blue curve vs our simulated blue curve!
import matplotlib.pyplot as plt
from qw_simulation import load_or_run_simulation

data = load_or_run_simulation()
s_unif_sim = data['s_unif'][:1001]

fig, ax = plt.subplots(figsize=(8, 4))
# Map x_coords and y_coords to physical t and sigma
# Let's see: x from roughly 60 to 450 corresponds to t from 0 to 1
# y from roughly 20 to 250 corresponds to sigma from 60 to 0
ax.plot(s_unif_sim, label='Simulated s_unif')
ax.set_title("Simulated sigma_unif(t)")
ax.set_ylim(25, 35)
plt.savefig("simulated_s_unif_zoom.png")
print("Saved simulated_s_unif_zoom.png")

