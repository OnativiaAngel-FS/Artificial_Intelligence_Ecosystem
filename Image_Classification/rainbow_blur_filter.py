from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import os
import numpy as np

def apply_blur_filter(image_path, output_path="rainbow_blurred.png"):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((128, 128))

        # Basic Gaussian blur first
        blurred = img.filter(ImageFilter.GaussianBlur(radius=2))

        # Convert to numpy for channel manipulation
        arr = np.array(blurred)

        # Split channels
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

        # Shift channels to create rainbow offset
        r_shifted = np.roll(r, shift=2, axis=1)   # right
        g_shifted = np.roll(g, shift=-2, axis=0)  # up
        b_shifted = np.roll(b, shift=2, axis=0)   # down

        rainbow = np.stack([r_shifted, g_shifted, b_shifted], axis=2)

        plt.imshow(rainbow)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Rainbow image saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    print("Image Blur Processor (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename (or 'exit' to quit): ").strip()
        if image_path.lower() == 'exit':
            print("Goodbye!")
            break
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue
        # derive output filename
        base, ext = os.path.splitext(image_path)
        output_file = f"{base}_rainbow_blurred{ext}"
        apply_blur_filter(image_path, output_file)