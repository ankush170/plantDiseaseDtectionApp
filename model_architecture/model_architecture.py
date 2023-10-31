import numpy as np
from tensorflow import ResNet50, Dense, GlobalAveragePooling2D, Model, Adam, EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

def build_and_train_model(train_generator, valid_generator, num_epochs, model_save_path):
    class_weights = compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)
    class_weight_dict = dict(enumerate(class_weights))

    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min')

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // valid_generator.batch_size,
        epochs=num_epochs,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint],
        class_weight=class_weight_dict
    )

    return model, history
