from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

def add_noise(image_path, noise_type='gaussian', sigma=20):
    # Open the image and convert to grayscale
    img = Image.open(image_path).convert('L')
    # Convert image to numpy array
    img_array = np.array(img)
    # Calculate image shape
    height, width = img_array.shape

    # Add noise to the image
    if noise_type == 'gaussian':
        # Generate Gaussian noise with given sigma
        noise = np.random.normal(0, sigma, (height, width))
        # Add noise to image array
        img_array = img_array + noise
        # Clip the pixel values between 0 and 255
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    elif noise_type == 'salt_and_pepper':
        # Generate Salt and Pepper noise
        noise = np.zeros((height, width))
        # Generate random indices for salt and pepper noise
        indices = np.random.choice((0, 1, 2), size=(height, width), p=[0.1, 0.1, 0.8])
        # Assign salt and pepper noise values
        noise[indices == 0] = 0
        noise[indices == 1] = 255
        # Add noise to image array
        img_array = img_array + noise
        # Clip the pixel values between 0 and 255
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # Create an image from the modified array and return it
    return Image.fromarray(img_array)

# Example usage
noisy_image = add_noise('data/data6.png', noise_type='gaussian', sigma=20)
plt.imshow(noisy_image, cmap='gray')
plt.savefig('data/data6n.png')
noisy_image.show()