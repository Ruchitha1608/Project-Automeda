"""
Generate sample histopathology image for demo
Creates a synthetic tissue-like image
"""
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import random

# Set random seed
random.seed(42)
np.random.seed(42)

# Image dimensions
width, height = 800, 600

# Create base image with tissue-like color (pinkish for H&E staining)
img = Image.new('RGB', (width, height), color=(245, 220, 230))

# Convert to numpy for manipulation
img_array = np.array(img)

# Add purple nuclei-like spots (hematoxylin staining)
draw = ImageDraw.Draw(img)
for _ in range(300):
    x = random.randint(0, width)
    y = random.randint(0, height)
    size = random.randint(3, 8)
    color = (random.randint(80, 120), random.randint(60, 100), random.randint(120, 160))
    draw.ellipse([x-size, y-size, x+size, y+size], fill=color)

# Add cellular structures
for _ in range(150):
    x = random.randint(0, width)
    y = random.randint(0, height)
    size = random.randint(10, 25)
    color = (random.randint(200, 240), random.randint(180, 220), random.randint(190, 230))
    draw.ellipse([x-size, y-size, x+size, y+size], fill=color, outline=(150, 120, 140))

# Add some darker regions (potential malignant areas)
for _ in range(5):
    x = random.randint(100, width-100)
    y = random.randint(100, height-100)
    size = random.randint(40, 80)
    color = (random.randint(180, 210), random.randint(150, 180), random.randint(170, 200))
    draw.ellipse([x-size, y-size, x+size, y+size], fill=color)
    
    # Add clustered nuclei in these regions
    for _ in range(20):
        nx = x + random.randint(-size, size)
        ny = y + random.randint(-size, size)
        nsize = random.randint(4, 7)
        ncolor = (random.randint(60, 90), random.randint(50, 80), random.randint(100, 130))
        draw.ellipse([nx-nsize, ny-nsize, nx+nsize, ny+nsize], fill=ncolor)

# Apply slight blur for more realistic tissue appearance
img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

# Add some noise
img_array = np.array(img)
noise = np.random.normal(0, 5, img_array.shape)
img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
img = Image.fromarray(img_array)

# Save
img.save('/Users/work/major proj/breast_cancer_ai/data/sample_breast.jpg', quality=95)

print("Generated sample histopathology image: 800x600 pixels")
