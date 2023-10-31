import numpy as np
from sklearn.metrics import classification_report


def calculate_classification_report(model, valid_generator):
    
    predictions = model.predict(valid_generator, steps=valid_generator.samples // valid_generator.batch_size, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = valid_generator.classes
    class_labels = list(valid_generator.class_indices.keys())

    
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    return report
