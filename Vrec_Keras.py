import keras
import numpy as np
import glob
import os
from skimage import io, transform
from keras.metrics import categorical_crossentropy as ccey
from keras.layers import Dropout, Dense, Flatten, Conv2D, MaxPooling2D


# resize to 100x100
w = 100
h = 100
c = 3


# load image
def read_img(path_):
    cate = [path_ + x_ for x_ in os.listdir(path_) if os.path.isdir(path_ + x_)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.png'):
            # print('reading the images:%s' % im)
            img = io.imread(im)
            img = transform.resize(img, (w, h))
            imgs.append(img)
            labels.append(idx)
        for im in glob.glob(folder + '/*.jpg'):
            # print('reading the images:%s' % im)
            img = io.imread(im)
            img = transform.resize(img, (w, h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


def load_1_img(path_):
    img = io.imread(path_)
    img = transform.resize(img, (w, h))
    return np.asarray(img, np.float32)


def get_y_train(y_train_, num_classes_):
    y_tmp = np.zeros([y_train_.shape[0], num_classes_])
    for i in range(y_train_.shape[0]):
        y_tmp[i][0], y_tmp[i][1] = (1, 0) if y_train_[i] == 0 else (0, 1)
    return y_tmp


def get_pred_result(x):
    return '洛天依' if x[0][0] > x[0][1] else '乐正绫'


num_classes = 2
# load images
train_path = '.\\dataset\\'
test_path = '.\\dataset.\\test\\'
x_train, y_train = read_img(train_path)
x_test, y_test = read_img(test_path)
y_train = get_y_train(y_train, num_classes)
y_test = get_y_train(y_test, num_classes)
img_rows, img_cols = w, h
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, -1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, -1)
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = keras.Sequential()
model.add(Conv2D(32, activation='relu', input_shape=(img_rows, img_cols, 3), nb_row=3, nb_col=3))
model.add(Conv2D(64, activation='relu', nb_row=3, nb_col=3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(.1))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(.3))
model.add(Dense(num_classes, activation='softmax'))

batch_size = 50
epochs = 100
adam = keras.optimizers.Adam(lr=0.001)
model.compile(loss=ccey, optimizer=adam, metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: %.5f\nTest accu: %.5f' % (score[0], score[1]))
model.save_weights('mymodel.h5')

model.load_weights('mymodel.h5')
print(get_pred_result(model.predict(load_1_img(r'.\rec\01.jpg').reshape([1, 100, 100, 3]))))
print(get_pred_result(model.predict(load_1_img(r'.\rec\02.jpg').reshape([1, 100, 100, 3]))))
print(get_pred_result(model.predict(load_1_img(r'.\rec\03.jpg').reshape([1, 100, 100, 3]))))
print(get_pred_result(model.predict(load_1_img(r'.\rec\04.jpg').reshape([1, 100, 100, 3]))))
print(get_pred_result(model.predict(load_1_img(r'.\rec\05.jpg').reshape([1, 100, 100, 3]))))
print(get_pred_result(model.predict(load_1_img(r'.\rec\06.jpg').reshape([1, 100, 100, 3]))))
print(get_pred_result(model.predict(load_1_img(r'.\rec\07.jpg').reshape([1, 100, 100, 3]))))
