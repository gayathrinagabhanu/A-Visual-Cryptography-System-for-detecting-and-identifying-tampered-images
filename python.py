import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.fft import dct
import pywt

# Function to divide image into blocks
def divide_image_into_blocks(image, block_size):
    blocks = []
    for i in range(0, image.shape[0], block_size):
        for j in range(0, image.shape[1], block_size):
            blocks.append(image[i:i + block_size, j:j + block_size])
    return blocks

def fwht(a):
    a = a.astype(np.int32)  # Cast to a larger integer type to avoid overflow
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a.astype(np.uint8)

def dwt_features(block):
    # Perform a single level 2D discrete wavelet transform on the block
    coeffs2 = pywt.dwt2(block, 'haar')
    LL, (LH, HL, HH) = coeffs2
    feature_vector = np.concatenate([LL.flatten(), LH.flatten(), HL.flatten(), HH.flatten()])
    return feature_vector

def extract_features(block):
    block_flattened = block.flatten()
    fwht_vec = fwht(block_flattened)
    lbp_vec = local_binary_pattern(block, P=8, R=1, method='uniform').flatten()
    dwt_vec = dwt_features(block)
    feature_vector = np.concatenate([fwht_vec, lbp_vec, dwt_vec])
    return feature_vector[:256]  # Ensure it matches block size

def embed_watermark(I, W):
    block_size = 16
    num_blocks = I.shape[0] // block_size
    if W.shape != I.shape:
        raise ValueError("Watermark W must be the same size as image I")

    F = np.zeros_like(I, dtype=np.float32)
    for i in range(num_blocks):
        for j in range(num_blocks):
            block = I[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            features = extract_features(block)
            F[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = features.reshape(block_size, block_size)

    F = (F - F.min()) / (F.max() - F.min()) * 255
    F = F.astype(np.uint8)
    K = np.random.randint(0, 256, I.shape, dtype=np.uint8)
    S = np.bitwise_xor(F, np.bitwise_xor(K, W))
    return F, K, S

def check_watermark(I, W, K):
    block_size = 16
    num_blocks = I.shape[0] // block_size
    F = np.zeros_like(I, dtype=np.float32)
    for i in range(num_blocks):
        for j in range(num_blocks):
            block = I[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            features = extract_features(block)
            F[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = features.reshape(block_size, block_size)

    reconstructed_W = np.bitwise_xor(F.astype(np.uint8), K)
    return np.array_equal(reconstructed_W, W)

# Function to process color images
def process_color_image(image, watermark, function, *args):
    channels = cv2.split(image)
    processed_channels = []
    for ch in channels:
        processed_channel = function(ch, watermark, *args)
        processed_channels.append(processed_channel)
    return cv2.merge(processed_channels)

# Example usage
I_color = cv2.imread('original.jpg')
W_color = cv2.imread('watermark image.jpg')

# Check if images are loaded correctly
if I_color is None:
    raise FileNotFoundError("Original image not found. Please check the path.")
if W_color is None:
    raise FileNotFoundError("Watermark image not found. Please check the path.")

# Ensure images are of size (256, 256)
I_color = cv2.resize(I_color, (256, 256))
W_color = cv2.resize(W_color, (256, 256))

# Convert watermark to grayscale to use the same watermark for all channels
W_gray = cv2.cvtColor(W_color, cv2.COLOR_BGR2GRAY)

# Embed the watermark in each channel
F_channels = []
K_channels = []
S_channels = []
for i in range(3):
    F, K, S = embed_watermark(I_color[:, :, i], W_gray)
    F_channels.append(F)
    K_channels.append(K)
    S_channels.append(S)

# Merge the watermarked channels
I_watermarked = cv2.merge([I_color[:, :, i] for i in range(3)])

# Save the watermarked image and watermark
cv2.imwrite('watermarked_image_color.png', I_watermarked)
cv2.imwrite('watermark_gray.png', W_gray)

# Function to recover the watermark
def recover_watermark(F_forged, S, K):
    return np.bitwise_xor(F_forged.astype(np.uint8), np.bitwise_xor(S, K))

# Function to detect forgery
def detect_forgery(FI, W, S, K):
    block_size = 16
    num_blocks = FI.shape[0] // block_size
    F_forged = np.zeros_like(FI, dtype=np.float32)
    for i in range(num_blocks):
        for j in range(num_blocks):
            block = FI[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            features = extract_features(block)
            F_forged[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = features.reshape(block_size, block_size)

    W_extracted = recover_watermark(F_forged, S, K)
    FRM = np.zeros((num_blocks, num_blocks), dtype=int)
    for i in range(num_blocks):
        for j in range(num_blocks):
            W_block = W[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            W_extracted_block = W_extracted[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            if np.array_equal(W_block, W_extracted_block):
                FRM[i, j] = 0  # Authentic block
            else:
                FRM[i, j] = 1  # Forged block

    return FRM

# Read forged image and shares
FI_color = cv2.imread('forged image.jpg')
if FI_color is None:
    raise FileNotFoundError("Forged image not found. Please check the path.")
FI_color = cv2.resize(FI_color, (256, 256))

FRM_channels = []
for i in range(3):
    FI_channel = FI_color[:, :, i]
    FRM = detect_forgery(FI_channel, W_gray, S_channels[i], K_channels[i])
    FRM_channels.append(FRM)

# Combine forgery detection results from all channels
FRM_combined = np.bitwise_or(np.bitwise_or(FRM_channels[0], FRM_channels[1]), FRM_channels[2])

if np.all(FRM_combined == 0):
    print('Image is Authentic and there is no forgery in the image')
else:
    print('Image is forged and the forged blocks are identified in the FRM')
    print(FRM_combined)
image1 = cv2.imread('original.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('watermarked image.jpg', cv2.IMREAD_GRAYSCALE)
image1 = cv2.resize(image1, (256, 256))
image2 = cv2.resize(image2, (256, 256))
# Compute SSIM between the two images
ssim_value, ssim_map = ssim(image1, image2, full=True)
psnr_value = psnr(image1, image2)

print(f"PSNR: {psnr_value} dB")
print(f"SSIM: {ssim_value}")


def bit_error_rate(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions for BER calculation")

    # Convert images to binary strings
    img1_bin = np.unpackbits(img1)
    img2_bin = np.unpackbits(img2)

    # Calculate the number of differing bits
    total_bits = img1_bin.size
    differing_bits = np.sum(img1_bin != img2_bin)

    # Compute BER
    ber = differing_bits / total_bits
    return ber
ber_value = bit_error_rate(image1,image2)

print(f"Bit Error Rate (BER): {ber_value}")
