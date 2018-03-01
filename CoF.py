import numpy as np
import matplotlib.pyplot as plt
import Kmeans
from scipy.ndimage import imread
from collections import Counter
import math
import scipy.misc as spm

def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))


def apply_kmeans(image_path):

    original_image = imread(image_path)
    #original_image = spm.imresize(original_image, (64, 64))
    original_image = np.array(original_image, dtype=np.float64) / 255
    w, h, d = original_shape = tuple(original_image.shape)

    assert d == 3

    quantizer,labels = Kmeans.Kmeans_algorithm(original_image,w,h,d)
    quantized_image = Kmeans.recreate_image(quantizer.cluster_centers_,labels,w,h)
    cluster_pixel_map = index = np.reshape(labels, (h,w))

    #plt.imshow(quantized_image)
    #plt.show()

    cluster_probability = Counter()
    cluster_probability.update(labels)
    cluster_probability = dict(cluster_probability)
    cluster_probability.update((k, float(v) / len(labels)) for k,v in cluster_probability.items())
       
    return original_image , quantized_image , cluster_pixel_map,cluster_probability

def collect_hard_cooccurrence(original_image , cluster_pixel_map,sigma_s,window_size = 5):

    cooccurrence_matrix = np.zeros([32,32])
    w,h,d = tuple(original_image.shape)
    width = window_size/2

    for i in range(0,w):
        for j in range(0,h):
            for k in range(0,window_size):
                for l in range(0,window_size):
                    neighbour_i = i - (width - k)
                    neighbour_j = j - (width - l)
                    if neighbour_i >= len(original_image):
                        neighbour_i -= len(original_image)
                    if neighbour_j >= len(original_image[0]):
                        neighbour_j -= len(original_image[0])
                    tau_a = cluster_pixel_map[i][j]
                    tau_b = cluster_pixel_map[neighbour_i][neighbour_j]
                    cooccurrence_matrix[tau_a][tau_b] += gaussian(distance(i,j,neighbour_i,neighbour_j),sigma_s)

    return cooccurrence_matrix

def hard2soft(co_hard,sigma_r,Z):
    
    cooccurrence_matrix = np.zeros([32,32])
    for tau_a in range(0,32):
        for tau_b in range(0,32):
            for k1 in range(0,32):
                for k2 in range(0,32):
                    cooccurrence_matrix[tau_a][tau_b] += ( (gaussian(tau_a - k1,sigma_r) * gaussian(tau_b - k2,sigma_r) * co_hard[k1][k2]) / Z * Z)

    return cooccurrence_matrix


def Cooc2PMI(co_soft,cluster_probability,n_colors=32):
    
    for i in range(0,n_colors):
        for j in range(0,n_colors):
            co_soft[i][j] = co_soft[i][j] / (cluster_probability[i] * cluster_probability[j])

    return co_soft

def apply_CoF(source,guidance_image,filtered_image, x, y, ws, sigma_s,M_T):

    hl = ws/2
    Wp = 0
    i_filtered = np.zeros([1,3])

    for i in range(0,ws):
        for j in range(0,ws):
            neighbour_x = x - (hl - i)
            neighbour_y = y - (hl - j)
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0]) 
                           
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            tau_a = guidance_image[x][y]
            tau_b = guidance_image[neighbour_x][neighbour_y]
            gi = M_T[tau_a][tau_b]
            
            w = gi * gs
            i_filtered += (source[neighbour_x][neighbour_y] * w)
            Wp += w

    i_filtered = i_filtered / Wp
    filtered_image[x][y] = i_filtered

def CoF(original_image,guidance_image,M_T,ws,sigma_s):

    filtered_image = np.zeros(original_image.shape)
    w , h , d = tuple(original_image.shape)
    for i in range(0,w):
        for j in range(0,h):
            apply_CoF(original_image,guidance_image,filtered_image,i,j,ws,sigma_s,M_T)

    return filtered_image

def CoF_util():

    image_path = 'lena.jpg'

    original_image , quantized_image , cluster_pixel_map,cluster_probability = apply_kmeans(image_path)
    guidance_image = cluster_pixel_map

    plt.figure(1)
    plt.clf()
    ax = plt.axes([0, 0, 1, 1])
    plt.axis('off')
    plt.title('Original image')
    plt.imshow(original_image) 
    plt.savefig('original.png')
         
    sigma_s = math.sqrt(2 * math.sqrt(15) + 1)
    sigma_r = 12
    Z = 2

    co_hard = collect_hard_cooccurrence(original_image,cluster_pixel_map,sigma_s)
    co_soft = hard2soft(co_hard,sigma_r,Z)
    M_T = Cooc2PMI(co_soft,cluster_probability)
    filtering_window_size = 3
    filtered_image = CoF(original_image,guidance_image,M_T,filtering_window_size,sigma_s)

    plt.figure(2)
    plt.clf()
    ax = plt.axes([0, 0, 1, 1])
    plt.axis('off')
    plt.title('Filtered_image')
    plt.imshow(filtered_image)
    plt.savefig('Fitered_image.png')
    plt.show()

CoF_util()
