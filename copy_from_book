# -*- codinng:utf-8 -*-
import os, shutil
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16




original_dataset_dir = r'E:\dataset'  # 原始数据集的地址
base_dir = r'C:\Users\dell\Desktop\smalldataset'  # 储存数据集的地址
os.mkdir(base_dir)
# copy images to training,validation,and test directories
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_good_dir = os.path.join(train_dir, 'good')
os.mkdir(train_good_dir)

train_bad_dir = os.path.join(train_dir, 'bad')
os.mkdir(train_bad_dir)

validation_good_dir = os.path.join(validation_dir, 'good')
os.mkdir(validation_good_dir)

validation_bad_dir = os.path.join(validation_dir, 'bad')
os.mkdir(validation_bad_dir)

test_good_dir = os.path.join(test_dir, 'good')
os.mkdir(test_good_dir)

test_bad_dir = os.path.join(test_dir, 'bad')
os.mkdir(test_bad_dir)

fnames = ['good.({}).jpg'.format(i) for i in range(500)]  # copy the first x good images to train_good_dir
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_good_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['good.({}).jpg'.format(i) for i in range(500,600)]  # copy the next y good images to validation_good_dir
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_good_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['good.({}).jpg'.format(i) for i in range(600, 688)]  # copy the next z good images to test_good_dir
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_good_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['bad.({}).jpg'.format(i) for i in range(800)]  # copy the first x bad images to train_bad_dir
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_bad_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['bad.({}).jpg'.format(i) for i in range(800, 1200)]  # copy the next y bad images to validation_bad_dir
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_bad_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['bad.({}).jpg'.format(i) for i in range(1200, 1300)]  # copy the next z bad images to test_bad_dir
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_bad_dir, fname)
    shutil.copyfile(src, dst)
# installing the VGG16 convolutional base

conv_base = VGG16(weights='imagenet',
                include_top=False,
                input_shape=(150, 150, 3))
# extractiong features using the pretrained convolutional base


base_dir = ' '
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_dorectory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size:(i + 1) * batch_size] = features_batch
        labels[i * batch_size:(i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


train_features, train_labels = extract_features(train_dir, 2 * x)
validation_features, validation_labels = extract_features(validation_dir, 2 * y)
test_features, test_labels = extract_features(test_dir, 2 * z)  # x,y,z为具体的图片的数字 依具体情况而定

train_features = np.reshape(train_features, (2 * x, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (2 * y, 4 * 4 * 512))
test_features = np.reshape(test_features, (2 * z, 4 * 4 * 512))

# defining and training the densely connected dlassidier

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))
# plotting the results 结果图表显示

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
