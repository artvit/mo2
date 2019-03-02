import os
import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


folder = '../data/'
dataset_small = 'notMNIST_small/'
dataset_large = 'notMNIST_large/'


def read_image_to_array(img_file):
    return cv2.imread(img_file)


def show_image(img_file):
    image = read_image_to_array(img_file)
    plt.imshow(image)
    plt.show()


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


def get_result_vector(j):
    e = np.zeros((len(index_dict), 1))
    e[index_dict[j]] = 1.0
    return e


def show_folder_stat(folder):
    print(folder)
    for char_folder in os.listdir(folder):
        directory = folder + '/' + char_folder
        files_number = len(os.listdir(directory))
        print(f'Class: {char_folder} Number of samples: {files_number}')


def get_all_data(dataset_folder):
    data, results = [], []
    for char_folder in os.listdir(dataset_folder):
        directory = dataset_folder + '/' + char_folder
        folder_common_result = get_result_vector(char_folder)
        filenames = os.listdir(directory)
        class_data, class_result = [], []
        for filename in tqdm(filenames, char_folder):
            class_data.append(read_image_to_array(directory + '/' + filename))
            class_result.append(folder_common_result.copy())
        data.extend(class_data)
        results.extend(class_result)
    print(f'Data set size: {len(data)}')
    return data, results


def main():
    # show_folder_stat(folder + dataset_small)
    # show_folder_stat(folder + dataset_large)
    data, results = get_all_data(folder + dataset_small)
    x_train, x_test, y_train, y_test = train_test_split(data, results, test_size=0.1, random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

    # show_image(folder + dataset_small + 'J/MDEtMDEtMDAudHRm.png')


if __name__ == '__main__':
    main()
