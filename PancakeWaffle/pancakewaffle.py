import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the path to your training and testing image directories
train_dir = 'C:/Users/Patrick Anderson/Desktop/PancakeWaffle/Train'
test_dir = 'C:/Users/Patrick Anderson/Desktop/PancakeWaffle/Test'

# Define image data generators for data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,      # Normalize pixel values to [0, 1]
    rotation_range=20,   # Randomly rotate images by 20 degrees
    width_shift_range=0.2,   # Randomly shift images horizontally by 20% of the width
    height_shift_range=0.2,  # Randomly shift images vertically by 20% of the height
    shear_range=0.2,     # Apply shear transformation with a shear intensity of 0.2
    zoom_range=0.2,      # Randomly zoom images by 20%
    horizontal_flip=True,   # Randomly flip images horizontally
    fill_mode='nearest'   # Fill any potential missing pixels after transformations
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only normalize pixel values for the test set

# Set the batch size and image dimensions
batch_size = 32
img_height = 150
img_width = 150

# Create the image data generators for the training and testing datasets
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
model.fit(train_data, epochs=epochs)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_data)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Predict the classes of the test images
predictions = model.predict(test_data)
predicted_classes = np.round(predictions).flatten()

# Get the class labels for reference
class_labels = train_data.class_indices

# Print the predictions for each image in the test dataset
for i, image in enumerate(test_data.filenames):
    print("Image:", image)
    print("Prediction:", predicted_classes[i])
    print("Class Label:", list(class_labels.keys())[int(predicted_classes[i])])
    print()