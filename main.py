import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize to 0-1
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Reshape 28x28 to 784
x_train_flat = x_train.reshape(-1, 784)
x_test_flat = x_test.reshape(-1, 784)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build Deep Neural Network
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(LeakyReLU(0.1))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# Important fix: reshape for CNN-like augmentation
x_train_aug = x_train.reshape(-1, 28, 28, 1)
x_test_aug = x_test.reshape(-1, 28, 28, 1)
datagen.fit(x_train_aug)

# Train model
history = model.fit(
    datagen.flow(x_train_aug, y_train, batch_size=64),
    epochs=10,
    validation_data=(x_test_aug, y_test)
)

# Evaluate
loss, accuracy = model.evaluate(x_test_flat, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

# Predictions
y_pred = model.predict(x_test_flat)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

# Visualization
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Pred: {y_pred_classes[i]}\nTrue: {y_true[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
