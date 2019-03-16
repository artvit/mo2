import os
import os.path

import cv2
import numpy as np
from tqdm import tqdm
from random import shuffle


folder = '../data/'
dataset_small = 'notMNIST_small'
dataset_large = 'notMNIST_large'

dataset = dataset_small
dataset_folder = dataset + '/'
filename = dataset


def get_full_filename(dataset_name):
    return folder + dataset_name


def read_image_to_array(img_file):
    # try:
    #     return mpimg.imread(img_file)
    # except OSError:
    #     return None
    return cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)


index_dict = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
}


def get_result(j):
    return index_dict[j]


def get_all_data(dataset_name, size=None):
    dataset_folder = folder + dataset_name
    data, results = [], []
    for char_folder in os.listdir(dataset_folder):
        directory = dataset_folder + '/' + char_folder
        folder_common_result = get_result(char_folder)
        filenames = os.listdir(directory)
        class_data, class_result = [], []
        for filename in tqdm(filenames, char_folder):
            file_data = read_image_to_array(directory + '/' + filename)
            if file_data is None:
                continue
            file_data = file_data.flatten()
            class_data.append(file_data)
            class_result.append(folder_common_result)
        data.extend(class_data)
        results.extend(class_result)
        print(f'Class: {char_folder} Size: {len(class_data)}')
    print(f'Data set size: {len(data)}')
    return resize_data(np.array(data), np.array(results), size)


def show_folder_stat(folder):
    print(folder)
    for char_folder in os.listdir(folder):
        directory = folder + '/' + char_folder
        files_number = len(os.listdir(directory))
        print(f'Class: {char_folder} Number of samples: {files_number}')


def save_data(filename, data, results):
    np.savez(folder + filename, data, results)


def load_data(filename):
    content = np.load(folder + filename + '.npz')
    data = content[content.files[0]]
    results = content[content.files[1]]
    print('Data is loaded')
    return data, results


def filter_duplicates(data, results):
    datatype = data.dtype.name
    data_dict = {}
    for x, y in tqdm(zip(data, results), 'Filtering progress'):
        data_dict[x.tobytes()] = y

    filtered_data_results_list = list(data_dict.items())
    new_data = np.array([np.frombuffer(x[0], dtype=datatype) for x in filtered_data_results_list])
    new_results = np.array([x[1] for x in filtered_data_results_list])
    print(f'Filtered data size: {len(filtered_data_results_list)}')
    return new_data, new_results


def resize_data(data, results, size=None):
    if size is None:
        return data, results
    datatype = data.dtype.name
    zipped = list(zip(list(data), list(results)))
    shuffle(zipped)
    resized = zipped[:size]
    result = list(zip(*resized))
    return np.array(result[0], dtype=datatype), np.array(result[1])
