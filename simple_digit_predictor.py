import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize to 0-1
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Reshape for augmentation and model compatibility
x_train_aug = x_train.reshape(-1, 28, 28, 1)
x_test_aug = x_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build Deep Neural Network
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(512),
    LeakyReLU(0.1),
    Dropout(0.3),
    
    Dense(256, activation='relu'),
    Dropout(0.3),
    
    Dense(10, activation='softmax')
])

# Compile model using RMSprop optimizer
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(x_train_aug)

# Train model
history = model.fit(
    datagen.flow(x_train_aug, y_train, batch_size=64),
    epochs=10,
    validation_data=(x_test_aug, y_test)
)

# Evaluate on test data
loss, accuracy = model.evaluate(x_test_aug, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

# Predictions
y_pred = model.predict(x_test_aug)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Visualize predictions
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test_aug[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {y_pred_classes[i]}\nTrue: {y_true[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
