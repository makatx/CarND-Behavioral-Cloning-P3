import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Flatten, Dense, Activation, MaxPooling2D, Lambda, Cropping2D

###################------------create and train model---------#################
def LeNet(train_generator, validation_generator, train_len, val_len, savefile="model.h5"):
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=X_train[0].shape))
    model.add(Lambda(lambda x: (x/255.0)-0.5))
    model.add(Conv2D(nb_filter=6, nb_row=5, nb_col=5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(nb_filter=16, nb_row=5, nb_col=5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(train_generator, samples_per_epoch=train_len, 
                                    validation_data=validation_generator, 
                                    nb_val_samples=val_len, nb_epoch=3)
    
    model.save(savefile)
    return history

def dave2(train_generator, validation_generator, train_len, val_len, savefile="model.h5", loadfile=False):
    
    if loadfile:
        model = load_model(savefile)
        
    else:
        
        model = Sequential()
        model.add(Cropping2D(((50,20), (0,0)), input_shape=(160, 320, 3)))
        model.add(Lambda((lambda x: (x/255.0)-0.5)))
        
        model.add(Conv2D(nb_filter=24, nb_col=5, nb_row=5, subsample=(2,2), activation='relu'))
        model.add(Conv2D(nb_filter=36, nb_col=5, nb_row=5, subsample=(2,2), activation='relu'))
        model.add(Conv2D(nb_filter=48, nb_col=5, nb_row=5, subsample=(2,2), activation='relu'))
        model.add(Conv2D(nb_filter=64, nb_col=3, nb_row=3, activation='relu'))
        model.add(Conv2D(nb_filter=64, nb_col=3, nb_row=3, activation='relu'))
        
        model.add(Flatten())
        model.add(Dense(1164, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))
        
        model.compile(loss='mse', optimizer='adam')
    
    history = model.fit_generator(train_generator, samples_per_epoch=train_len, 
                                    validation_data=validation_generator, 
                                    nb_val_samples=val_len, nb_epoch=5)
    
    model.save(savefile)
    return history

def plot_history(history_object):
    print(history_object.history.keys())
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def generator(samples, batch_size=128):
    num_samples = len(samples)
    batch_size = int(batch_size/2)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Remote Driving')
    
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='training_images',
        help='Path to image folder. This is where the images for training are stored.'
    )
    args = parser.parse_args()
    
    
image_folder = args.image_folder + '\\'

lines = []
image_list = []
steer_angle = []
with open(image_folder + 'driving_log.csv', 'r') as f :
    driving_log = csv.reader(f)
    for row in driving_log:
        lines.append(row)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

history = dave2(train_generator, validation_generator, len(train_samples)*2, 
                len(validation_samples)*2, savefile="model.h5", loadfile=False)
plot_history(history)
