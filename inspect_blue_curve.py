from PIL import Image
import numpy as np

img_b = Image.open('orig_panel_b.png')
arr_b = np.array(img_b)

print("Image shape:", arr_b.shape)

# Let's inspect column x=150 (which corresponds to j=-15) in panel b:
col = arr_b[:, 150]
# Find where R < 50 and B > 150
blue_pixels = np.where((col[:, 0] < 50) & (col[:, 2] > 150))[0]
print("Blue pixel y-indices in col 150:", blue_pixels)
for y in blue_pixels:
    print(f"y={y}: RGB={col[y][:3]}")

