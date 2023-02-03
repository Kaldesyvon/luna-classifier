import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def main():
    Categories = ['Luna']
    flat_data_arr = []  # input array
    target_arr = []  # output array
    datadir = './luna_imgs/'
    # path which contains all the categories of images
    for i in Categories:
        print(f'loading... category : {i}')
        path = os.path.join(datadir, i)
        for img in os.listdir(path):
            img_array = imread(os.path.join(path, img))
            img_resized = resize(img_array, (150, 150, 3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(Categories.index(i))
            print(f'{img} image done...')
        print(f'loaded category:{i} successfully')
    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)
    df = pd.DataFrame(flat_data)  # dataframe
    df['Target'] = target
    x = df.iloc[:, :-1]  # input data
    y = df.iloc[:, -1]  # output data

    svc = svm.SVC(kernel='poly', gamma=0.005, degree=2, coef0=0, C=0.7, probability=True)

    svc.fit(x, y)

    while 1:
        url = input('Enter URL of Image :')
        img = imread(url)
        plt.imshow(img)
        plt.show()
        img_resize = resize(img, (150, 150, 3))
        l = [img_resize.flatten()]
        probability = svc.predict_proba(l)
        for ind, val in enumerate(Categories):
            print(f'{val} = {probability[0][ind] * 100}%')
        print("The predicted image is : " + Categories[svc.predict(l)[0]])

        print('not valid url')


if __name__ == '__main__':
    main()
