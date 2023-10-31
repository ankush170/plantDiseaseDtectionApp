from data_processing import load_and_preprocess_data
from model_architecture import build_and_train_model
from results import calculate_classification_report


dataset_dir = "/Users/anhappy170/Desktop/Dataset"
image_size = (224, 224)
batch_size = 32
num_epochs = 20


model_save_path = "final_model.h5"


train_generator, valid_generator = load_and_preprocess_data(dataset_dir, image_size, batch_size)


model, history = build_and_train_model(train_generator, valid_generator, num_epochs, model_save_path)


classification_report = calculate_classification_report(model, valid_generator)
print(classification_report)


model.save("final_model.h5")
