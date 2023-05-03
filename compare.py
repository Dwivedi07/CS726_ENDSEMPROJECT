import cv2
from skimage.metrics import structural_similarity as ssim

def compare():
    # Load the actual and denoised images
    actual_img = cv2.imread(actual_img)
    denoised_img = cv2.imread(denoised_img)
    # actual_img = cv2.imread('data/im3.png')
    # denoised_img = cv2.imread('output/denoise_image.png')

    # Resize the images to a common size
    actual_img = cv2.resize(actual_img, (500, 500))
    denoised_img = cv2.resize(denoised_img, (500, 500))

    # Convert the images to grayscale
    actual_gray = cv2.cvtColor(actual_img, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)

    # Calculate the SSIM score between the images
    ssim_score = ssim(actual_gray, denoised_gray, multichannel=False)

    # Print the SSIM score
    file = open(SSIMscore, 'w')
    for element in range(6):
        file.writelines(f'SSIM score for data {element}: {ssim_score}')
        file.write('\n')
    file.close()
    # print("SSIM score between the actual and denoised images: {:.2f}".format(ssim_score))