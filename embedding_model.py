import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

ROOT = 'data'
CLASS_NAME = sorted(list(os.listdir(f'{ROOT}/train')))

embedding_function = OpenCLIPEmbeddingFunction()


def get_single_image_embedding(image):
    # This function get embedding values of each image
    embedding = embedding_function._encode_image(image=image)
    return np.array(embedding)


def read_image_from_path(path, size):
    # Input: path to image and standard size of image
    img = Image.open(path).convert('RGB').resize(size)
    return np.array(img)


def folder_to_images(folder, size):
    list_dir = [folder + '/' + name for name in os.listdir(folder)]
    images_np = np.zeros(shape=(len(list_dir), *size, 3))
    images_path = []
    for i, path in enumerate(list_dir):
        images_np[i] = read_image_from_path(path, size)
        images_path.append(path)

    images_path = np.array(images_path)
    return images_np, images_path


def plot_results(querquery_pathy, ls_path_score, reverse):
    fig = plt.figure(figsize=(15, 9))
    fig.add_subplot(2, 3, 1)
    plt.imshow(read_image_from_path(querquery_pathy, size=(448, 448)))
    plt.title(f"Query Image: {querquery_pathy.split('/')[2]}", fontsize=16)
    plt.axis("off")
    for i, path in enumerate(sorted(ls_path_score, key=lambda x: x[1], reverse=reverse)[:5], 2):
        fig.add_subplot(2, 3, i)
        plt.imshow(read_image_from_path(path[0], size=(448, 448)))
        plt.title(f"Top {i-1}: {path[0].split('/')[2]}", fontsize=16)
        plt.axis("off")
    plt.show()


def absolute_difference(query, data):
    # Calculate the abs difference between data and query
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.sum(np.abs(data - query), axis=axis_batch_size)


def get_l1_score(root_img_path, query_path, size):
    # This function computes and return query image and difference rate
    query = read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size)
            embedding_lst = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(
                    images_np[idx_img].astype(np.uint8))
                embedding_lst.append(embedding)
            rates = absolute_difference(
                query_embedding, np.stack(embedding_lst))
            ls_path_score.extend(list(zip(images_path, rates)))

    return query, ls_path_score


def mean_square_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.mean((data - query)**2, axis=axis_batch_size)


def get_l2_score(root_img_path, query_path, size):
    # Mostly the same to l1 except for the rate now becomes the MSE
    query = read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(
                path, size)
            embedding_lst = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(
                    images_np[idx_img].astype(np.uint8))
                embedding_lst.append(embedding)
            rates = mean_square_difference(
                query_embedding, np.stack(embedding_lst))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


def cosine_similarity(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_norm = np.sqrt(np.sum(query**2))
    data_norm = np.sqrt(np.sum(data**2, axis=axis_batch_size))
    return np.sum(data * query, axis=axis_batch_size) / (query_norm*data_norm + np.finfo(float).eps)


def get_cosine_similarity_score(root_img_path, query_path, size):
    # Mostly the same to l1 and l2 except for the rate now becomes the Cosine Simlarity
    query = read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(
                path, size)
            embedding_lst = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(
                    images_np[idx_img].astype(np.uint8))
                embedding_lst.append(embedding)
            rates = cosine_similarity(query_embedding, np.stack(embedding_lst))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


def correlation_coefficient(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_mean = query - np.mean(query)
    data_mean = data - np.mean(data, axis=axis_batch_size, keepdims=True)
    query_norm = np.sqrt(np.sum(query_mean**2))
    data_norm = np.sqrt(np.sum(data_mean**2, axis=axis_batch_size))

    return np.sum(data_mean * query_mean, axis=axis_batch_size) / (query_norm*data_norm + np.finfo(float).eps)


def get_correlation_coefficient_score(root_img_path, query_path, size):
    # Mostly the same to the rest except for the rate now becomes the Correlation Coefficient
    query = read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(
                path, size)
            embedding_lst = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(
                    images_np[idx_img].astype(np.uint8))
                embedding_lst.append(embedding)
            rates = correlation_coefficient(
                query_embedding, np.stack(embedding_lst))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


# Test case
root_img_path = f"{ROOT}/train/"
# query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
query_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
size = (448, 448)
# query, ls_path_score = get_l1_score(root_img_path, query_path, size)
# query, ls_path_score = get_l2_score(root_img_path, query_path, size)
# query, ls_path_score = get_cosine_similarity_score(
#     root_img_path, query_path, size)
query, ls_path_score = get_correlation_coefficient_score(
    root_img_path, query_path, size)
# reverse=True for cosine and corr_coef, False for the remaining
plot_results(query_path, ls_path_score, reverse=True)
