import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def mean_filter(image, kernel_size=3):
    """
    Apply mean filter to the input image.
    
    Parameters:
    image (numpy.ndarray): Input image array.
    kernel_size (int): Size of the mean filter kernel.
    
    Returns:
    numpy.ndarray: Denoised image.
    """
    # Convert image to float32 for better precision
    image = image.astype(np.float32)
    
    # Define the kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    
    # Get the image dimensions
    height, width, channels = image.shape
    
    # Initialize the output image
    denoised_image = np.zeros((height, width, channels), dtype=np.float32)
    
    # Pad the image to handle borders
    pad_size = kernel_size // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    
    # Apply the mean filter
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                region = padded_image[y:y + kernel_size, x:x + kernel_size, c]
                denoised_image[y, x, c] = np.sum(region * kernel)
    
    # Clip the values to be in the valid range
    denoised_image = np.clip(denoised_image, 0, 255)
    
    # Convert back to uint8
    denoised_image = denoised_image.astype(np.uint8)
    
    return denoised_image

def calculate_psnr(original, denoised):
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between two images.
    
    Parameters:
    original (numpy.ndarray): Original image array.
    denoised (numpy.ndarray): Denoised image array.
    
    Returns:
    float: PSNR value.
    """
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def denoise_image(input_image_path, output_image_path, kernel_size=3):
    # Read the image
    image = np.array(Image.open(input_image_path))
    
    # Apply the mean filter
    denoised_image = mean_filter(image, kernel_size)
    
    # Save the denoised image
    denoised_image_pil = Image.fromarray(denoised_image)
    denoised_image_pil.save(output_image_path)
    
    # Display the original and denoised images using matplotlib
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Denoised Image")
    plt.imshow(denoised_image)
    plt.axis('off')
    
    plt.show()
    
    # Calculate PSNR
    psnr_value = calculate_psnr(image, denoised_image)
    print(f'PSNR Value: {psnr_value:.2f} dB')

# Example usage
input_image_path = 'Figure-21-original-image-and-noisy-images-a-Original-image-without-noise-b-Image.jpg'  # Path to the input image
output_image_path = 'output.jpg'  # Path to save the denoised image
denoise_image(input_image_path, output_image_path)
