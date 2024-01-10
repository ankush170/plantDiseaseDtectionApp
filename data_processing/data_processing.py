import numpy as np
from tensorflow import ImageDataGenerator

def load_and_preprocess_data(dataset_dir, image_size, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
        brightness_range=[0.8, 1.2],  # Adjust brightness
        channel_shift_range=10,  # Random channel shifts
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    valid_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, valid_generator
