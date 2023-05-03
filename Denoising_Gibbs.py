import math
import numpy as np
import matplotlib.pyplot as plt
import argparse, time
import os
import cv2
from skimage.metrics import structural_similarity as ssim

def load_image(filename):
    my_img = plt.imread(filename)
    img_gray = np.dot(my_img[..., :3], [0.2989, 0.5870, 0.1140])
    img_gray = np.where(img_gray > 0.5, 1, -1)
    img_padded = np.zeros([img_gray.shape[0] + 2, img_gray.shape[1] + 2])
    img_padded[1:-1, 1:-1] = img_gray
    return img_padded


def sample_y(i, j, Y, X):
    markov_blanket = [Y[i - 1, j], Y[i, j - 1], Y[i, j + 1], Y[i + 1, j], X[i, j]]
    w = ITA * markov_blanket[-1] + BETA * sum(markov_blanket[:4])
    prob = 1 / (1 + math.exp(-2*w))
    return (np.random.rand() < prob) * 2 - 1


def get_posterior(filename, burn_in_steps, total_samples, logfile):
    X = load_image(filename)
    posterior = np.zeros(X.shape)
    print(X.shape)
    Y = np.random.choice([1, -1], size=X.shape)
    energy_list = list()
    # prev_energy = 0
    # convergence_threshold = 0.001

    for step in range(burn_in_steps + total_samples):
        energy = -np.sum(np.multiply(Y, X))*ITA-(np.sum(np.multiply(Y[:-1], Y[1:]))+np.sum(np.multiply(Y[:, :-1], Y[:, 1:])))*BETA
        # print(f"Iteration: {step},{energy}")
        # if abs(energy - prev_energy) < convergence_threshold:
        #     break
        prev_energy = energy
        for i in range(1, Y.shape[0]-1):
            for j in range(1, Y.shape[1]-1):
                # print(f"Iteration: {step}, i: {i}, j: {j}, size: {Y.shape[0]}, {X.shape[0]}")
                y = sample_y(i, j, Y, X)
                Y[i, j] = y
                if y == 1 and step >= burn_in_steps:
                    posterior[i, j] += 1
        energy = -np.sum(np.multiply(Y, X))*ITA-(np.sum(np.multiply(Y[:-1], Y[1:]))+np.sum(np.multiply(Y[:, :-1], Y[:, 1:])))*BETA
        if step < burn_in_steps:
            energy_list.append(str(step) + "\t" + str(energy) + "\tB")
        else:
            energy_list.append(str(step) + "\t" + str(energy) + "\tS")
    posterior = posterior / total_samples

    file = open(logfile, 'w')
    for element in energy_list:
        file.writelines(element)
        file.write('\n')
    file.close()
    return posterior


def denoise_image(filename, burn_in_steps, total_samples, logfile):
    posterior = get_posterior(filename, burn_in_steps, total_samples, logfile=logfile)
    denoised = np.zeros(posterior.shape, dtype=np.float64)
    denoised[posterior > 0.5] = 1
    return denoised[1:-1, 1:-1]


def plot_energy(filename,i):
    x = np.genfromtxt(filename, dtype=None, encoding='utf8')
    its, energies, phases = zip(*x)
    its = np.asarray(its)
    energies = np.asarray(energies)
    phases = np.asarray(phases)
    burn_mask = (phases == 'B')
    samp_mask = (phases == 'S')
    assert np.sum(burn_mask) + np.sum(samp_mask) == len(x), 'Found bad phase'
    its_burn, energies_burn = its[burn_mask], energies[burn_mask]
    its_samp, energies_samp = its[samp_mask], energies[samp_mask]
    p1, = plt.plot(its_burn, energies_burn, 'r')
    p2, = plt.plot(its_samp, energies_samp, 'b')
    plt.title("energy")
    plt.xlabel('iteration number')
    plt.ylabel('energy')
    plt.legend([p1, p2], ['burn in', 'sampling'])
    plt.savefig('%s.png' % filename)
    plt.close()


def save_image(denoised_image,i):
    plt.imshow(denoised_image, cmap='gray')
    plt.title("denoised image")
    plt.savefig(f'output_GIBBS/data{i}dn.png')
    plt.close()



def compare(a,dn,i):
    # Load the actual and denoised images
    actual_img = cv2.imread(a)
    denoised_img = cv2.imread(dn)
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
    file.write('\n')
    file.writelines(f'SSIM score for data {i}: {ssim_score}')
    file.write('\n')
    file.close()
    print("SSIM score between the actual and denoised images: {:.2f}".format(ssim_score))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=False, help='The task to run. Valid values are: 1, 2, 3, all')
    args = parser.parse_args()
    i = args.data
    ITA = 1
    BETA = 1
    total_samples = 900
    burn_in_steps = 100

    
    SSIMscore = f"SSIMscor_gibbs"

    # logfile = "output_GIBBS/log_energy"
    # denoised_img = denoise_image(f"data/data{i}.png", burn_in_steps=burn_in_steps,
    #                             total_samples=total_samples, logfile=logfile)
    # plot_energy(logfile,i)
    # os.rename("output_GIBBS/log_energy",f'output_GIBBS/log_energy{i}')
    # save_image(denoised_img,i)
    # compare(f'data/data{i}.png',f'output_GIBBS/data{i}dn.png')
    
    for i in range(1,7):
        logfile = f"output_GIBBS/log_energy{i}"
        denoised_img = denoise_image(f"data/data{i}.png", burn_in_steps=burn_in_steps,
                                    total_samples=total_samples, logfile=logfile)
        plot_energy(logfile,i)
        # os.rename("output_GIBBS/log_energy",f'output_GIBBS/log_energy{i}')
        save_image(denoised_img,i)
        compare(f'data/data{i}.png',f'output_GIBBS/data{i}dn.png',i)