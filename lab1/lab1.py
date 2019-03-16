import os
import os.path

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import dataloader


def show_image(img_file):
    image = dataloader.read_image_to_array(img_file)
    plt.imshow(image, cmap='gray')
    plt.show()


def scale_data(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)


def main():
    # show_folder_stat(folder + dataset_small)
    # show_folder_stat(folder + dataset_large)

    def save_data_to_file(dataset_name):
        data, results = dataloader.get_all_data(dataset_name)
        data, results = dataloader.filter_duplicates(data, results)
        dataloader.save_data(dataset_name, data, results)

    # save_data_to_file(dataloader.dataset_small)
    # save_data_to_file(dataloader.dataset_large)

    data, results = dataloader.load_data(dataloader.dataset_small)
    # data, results = dataloader.resize_data(data, results, 100)
    # data = scale_data(data)

    x_train, x_test, y_train, y_test = train_test_split(data, results, test_size=0.1, random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

    logreg = LogisticRegression(
        solver='lbfgs',
        multi_class='multinomial',
        # verbose=10,
        # max_iter=20
    )
    logreg.fit(x_train, y_train)
    # y_pred = logreg.predict(x_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f'Test set Accuracy: {accuracy}')
    test_score = logreg.score(x_test, y_test)
    print(f'Test set Score: {test_score}')
    test_score = logreg.score(x_valid, y_valid)
    print(f'Validation set Score: {test_score}')

    # show_image(folder + dataset_small + 'A/RGVtb2NyYXRpY2FCb2xkT2xkc3R5bGUgQm9sZC50dGY=.png')
    # show_image(folder + dataset_small + 'J/MDEtMDEtMDAudHRm.png')


if __name__ == '__main__':
    main()
