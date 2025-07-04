import os
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

# ✅ Define constants
batch_size = 128
num_classes = 10
epochs = 10
input_shape = (28, 28, 1)

# 📥 Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape to fit Keras' expected input
x_train = x_train.reshape(-1, *input_shape).astype("float32") / 255.0
x_test = x_test.reshape(-1, *input_shape).astype("float32") / 255.0

# Convert class labels to one-hot encoding
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 🧠 Build the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# ⚙️ Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 🚀 Train the model
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test)
)

print("\n✅ The model has successfully trained!")

# 💾 Save the model (recommended format)
os.makedirs("model", exist_ok=True)
model.save("model/mnist_model.keras")
print("📁 Model saved as: model/mnist_model.keras")
