import numpy as np
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

def build_and_train_model(train_generator, valid_generator, num_epochs, model_save_path):
    class_weights = compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)
    class_weight_dict = dict(enumerate(class_weights))

    # Training using ResNet50 for 10 epochs
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min')

    history_resnet = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // valid_generator.batch_size,
        epochs=10,  # Training for 10 epochs with ResNet50
        verbose=1,
        callbacks=[early_stopping, model_checkpoint],
        class_weight=class_weight_dict
    )

    # Training using DenseNet121 for the next 10 epochs
    base_model = DenseNet121(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    history_densenet = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // valid_generator.batch_size,
        epochs=10,  # Training for 10 epochs with DenseNet121
        verbose=1,
        callbacks=[early_stopping, model_checkpoint],
        class_weight=class_weight_dict
    )

    # Returning both histories for analysis if needed
    return model, history_resnet, history_densenet
