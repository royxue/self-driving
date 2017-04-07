import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Lambda, Dense, Cropping2D, Activation
from keras.utils import plot_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


lines = []
with open('./train/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for l in reader:
        lines.append(l)

train, valid = train_test_split(lines, test_size=0.2)

def data_gen(lines, batch_size=64):
    num_lines = len(lines)
    while 1:
        shuffle(lines)
        for offset in range(0, num_lines, batch_size):
            batch_lines = lines[offset: offset+batch_size]

            images = []
            angles = []

            for l in batch_lines:
                c_path = './train/IMG/' + l[0].split('/')[-1]
                l_path = './train/IMG/' + l[1].split('/')[-1]
                r_path = './train/IMG/' + l[2].split('/')[-1]
                c_image = cv2.imread(c_path)
                l_image = cv2.imread(l_path)
                r_image = cv2.imread(r_path)

                angle = float(l[3])
                l_angle = angle + 0.25
                r_angle = angle - 0.25
                images.append(c_image)
                images.append(cv2.flip(c_image, 1))
                images.append(l_image)
                images.append(r_image)
                angles.append(angle)
                angles.append(-angle)
                angles.append(l_angle)
                angles.append(r_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def bcnet():
    NUM_BATCH = 64
    EPOCHS = 5

    train_gen = data_gen(train, NUM_BATCH)
    valid_gen = data_gen(valid, NUM_BATCH)

    model = Sequential()
    model.add(Lambda(lambda x:(x/127.5)-1.0, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 20), (0, 0))))
    # Conv1
    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    
    # Conv2
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))

    #Conv3
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))

    # Conv4
    model.add(Conv2D(512, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    
    # Conv5
    model.add(Conv2D(512, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(512, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    
    # FC
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    train_steps = len(train)/NUM_BATCH
    valid_steps = len(valid)/NUM_BATCH
    metrics = model.fit_generator(train_gen, steps_per_epoch=train_steps, validation_data=valid_gen, validation_steps=valid_steps, epochs=EPOCHS, verbose=1)

    print(metrics.history['loss'])
    print(metrics.history['val_loss'])

    model.save('bcnet.h5')
    print(metrics.history['loss'])
    print(metrics.history['val_loss'])

def nvidianet():
    NUM_BATCH = 64
    EPOCHS = 8

    train_gen = data_gen(train, NUM_BATCH)
    valid_gen = data_gen(valid, NUM_BATCH)

    model = Sequential()
    model.add(Lambda(lambda x:(x/127.5)-1.0, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 20), (0, 0))))
    # Conv1
    model.add(Conv2D(24,(5,5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))

    # Conv2
    model.add(Conv2D(36, (5,5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))

    #Conv3
    model.add(Conv2D(48, (5,5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))

    # Conv4
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))

    # FC
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))

    plot_model(model, to_file='model.png')

    model.compile(loss='mse', optimizer='adam')
    train_steps = len(train)/NUM_BATCH
    valid_steps = len(valid)/NUM_BATCH
    metrics = model.fit_generator(train_gen, steps_per_epoch=train_steps, validation_data=valid_gen, validation_steps=valid_steps, epochs=EPOCHS, verbose=1)

    print(metrics.history['loss'])
    print(metrics.history['val_loss'])

    model.save('nvidianet.h5')

if __name__ == "__main__":
    nvidianet()
