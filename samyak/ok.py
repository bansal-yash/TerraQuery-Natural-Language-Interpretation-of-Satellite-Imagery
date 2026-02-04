from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Load the original image
image_path = 'P0003_0002.png'  # Replace with the actual path to your image
img = Image.open(image_path)

W, H = img.size  # Actual image width and height
scale_x = W / 1000
scale_y = H / 1000

draw = ImageDraw.Draw(img)

# Bounding boxes normalized to 1000
bboxes = [
    (10, 442, 120, 636),   # Top row, leftmost
    (110, 436, 220, 630),  # Top row, second
    (198, 430, 308, 624),  # Top row, third
    (286, 424, 396, 618),  # Top row, fourth
    (0, 928, 90, 1000),    # Bottom row, left
    (100, 958, 220, 1000)  # Bottom row, right
]

# Draw rectangles using scaled coordinates
for x1, y1, x2, y2 in bboxes:
    sx1 = x1 * scale_x
    sy1 = y1 * scale_y
    sx2 = x2 * scale_x
    sy2 = y2 * scale_y
    draw.rectangle([sx1, sy1, sx2, sy2], outline="red", width=4)

# Display the image
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.title("Yellow Buses with Bounding Boxes")
plt.show()

# Save the output
img.save('annotated_buses.jpg')
print("✅ Annotated image saved as 'annotated_buses.jpg'")
