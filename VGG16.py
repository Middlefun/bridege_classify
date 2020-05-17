# -*- codinng:utf-8 -*-
import os, shutil
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import models
from keras.layers import Conv2D, Dense, Dropout, MaxPool2D, Flatten
from keras import optimizers
from keras.applications import VGG16
from keras.optimizers import SGD

ORIGINAL_DATASET_DIR= r'/home/deeplearning/classify_pictures/classified1'  # 原始数据集的地址
DESTINATION_DIR = r'/home/deeplearning/tmp_store_for_classify'  # 储存数据集的地址
REFRESH_FLAG = False     # true: refresh the data; false: use the old data
BATCH_SIZE = 256
EPOCH = 30
STEPS_PER_EPOCH = 10
MODEL_PATH = r'/home/deeplearning/PycharmProjects/location_classify/model_0001'


def refresh_data(
        original_dir, validation_split, test_split,
        destination_dir=DESTINATION_DIR):
    """copy files from original_path to destination_path, refresh the data for flow_from_directory.

    Args:
        original_dir: the path which save the pictures of the same type.
        validation_split: float from 0 to 1, which means how many rate of the data for validation.
        test_split: float from 0 to 1, which means how many rate of the data for test.
        destination_dir: the path to save the pictures.

    Returns:

    """
    types = os.listdir(original_dir)

    # create the path
    if os.path.exists(destination_dir):
        print("refreshing the data....................")
        shutil.rmtree(destination_dir)
    os.mkdir(destination_dir)

    # create the train, validation, test folders
    for folder in ['train', 'validation', 'test']:
        new_path = os.path.join(destination_dir, folder)
        os.mkdir(new_path)

    for bridge_type in types:
        print('preparing for {} .........'.format(bridge_type))
        type_path = os.path.join(original_dir, bridge_type)
        train_data = os.listdir(type_path)
        validation_data = []
        test_data = []
        data_number = len(train_data)

        # randomly pick the picture into validation data
        for i in range(round(data_number * validation_split)):
            index = np.random.randint(0, len(train_data))
            validation_data.append(train_data[index])
            train_data.pop(index)

        # randomly pick the picture into test data
        for j in range(round(data_number * test_split)):
            index = np.random.randint(0, len(train_data))
            test_data.append(train_data[index])
            train_data.pop(index)

        for real_path, picture_data in zip(['train', 'validation', 'test'],
                                       [train_data, validation_data, test_data]):
            real_path = os.path.join(destination_dir, real_path, bridge_type)
            os.mkdir(real_path)
            for fname in picture_data:
                src = os.path.join(original_dir, bridge_type, fname)
                dst = os.path.join(real_path, fname)
                shutil.copyfile(src, dst)


if __name__ == '__main__':
    if REFRESH_FLAG:
        refresh_data(ORIGINAL_DATASET_DIR, 0.2, 0)

    train_dir = os.path.join(DESTINATION_DIR, 'train')
    validation_dir = os.path.join(DESTINATION_DIR, 'validation')
    test_dir = os.path.join(DESTINATION_DIR, 'test')

    # create data generator
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(150, 150), batch_size=BATCH_SIZE
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir, target_size=(150, 150), batch_size=BATCH_SIZE
    )

    # ============================================================
    # create the model
    # ============================================================
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    conv_base.trainable = False # freeze the vgg16 base

    my_model = models.Sequential()
    my_model.add(conv_base)
    my_model.add(Flatten())
    my_model.add(Dense(256, activation='relu'))
    my_model.add(Dense(3, activation='softmax'))

    my_model.summary()      # show the model

    my_model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=2e-5), metrics=['accuracy'])
    """model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
    """
    history = my_model.fit_generator(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCH,
        validation_data=validation_generator,
        validation_steps=STEPS_PER_EPOCH
    )
    my_model.save(MODEL_PATH)

    # =============================================
    # plotting the results 结果图表显示
    # =============================================
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
