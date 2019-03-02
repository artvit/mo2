import os
import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

folder = '../data/'
dataset_small = 'notMNIST_small'
dataset_large = 'notMNIST_large'

dataset = dataset_small
dataset_folder = dataset + '/'
filename = dataset


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


def get_result(j):
    return index_dict[j]


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
    return np.array(data), np.array(results)


def save_data(filename, data, results):
    np.savez(filename, data, results)


def load_data(filename):
    content = np.load(filename + '.npz')
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


def main():
    # show_folder_stat(folder + dataset_small)
    # show_folder_stat(folder + dataset_large)

    # data, results = get_all_data(folder + dataset_folder)
    # data, results = filter_duplicates(data, results)
    # save_data(filename, data, results)

    data, results = load_data(filename)

    x_train, x_test, y_train, y_test = train_test_split(data, results, test_size=0.1, random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

    logreg = LogisticRegression(C=1e2, max_iter=1000, solver='lbfgs', multi_class='multinomial')
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # show_image(folder + dataset_small + 'A/RGVtb2NyYXRpY2FCb2xkT2xkc3R5bGUgQm9sZC50dGY=.png')
    # show_image(folder + dataset_small + 'J/MDEtMDEtMDAudHRm.png')


if __name__ == '__main__':
    main()
