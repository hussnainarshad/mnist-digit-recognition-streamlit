import streamlit as st
from tensorflow.keras.models import load_model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load model and data
@st.cache_resource
def load_mnist_model_and_data():
    model = load_model("model/mnist_model.keras")
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    return model, x_test, y_test

model, x_test, y_test = load_mnist_model_and_data()

# Streamlit app
st.title("MNIST Digit Prediction")

st.write("""
This app shows the model's prediction for digits in the MNIST test set.
Select any test sample to see the prediction.
""")

# Create digit selection interface
max_index = len(x_test) - 1
col1, col2 = st.columns(2)

with col1:
    if st.button("Random Sample"):
        random_index = np.random.randint(0, max_index)
        st.session_state.selected_index = random_index

with col2:
    selected_index = st.number_input(
        f"Enter sample index (0-{max_index})",
        min_value=0,
        max_value=max_index,
        value=0,
        step=1,
        key="selected_index"
    )

# Display the selected sample
st.subheader(f"Test Sample #{selected_index}")
st.write(f"Actual Digit: {y_test[selected_index]}")

fig, ax = plt.subplots()
ax.imshow(x_test[selected_index].reshape(28, 28), cmap='gray')
ax.axis('off')
st.pyplot(fig)

# Get prediction
prediction = model.predict(x_test[selected_index:selected_index+1])
predicted_label = np.argmax(prediction, axis=1)[0]
actual_label = y_test[selected_index]

# Display results
st.subheader("Prediction Results")
if predicted_label == actual_label:
    st.success(f"✅ Model prediction: {predicted_label}")
else:
    st.error(f"❌ Model prediction: {predicted_label}")
st.write(f"Actual digit: {actual_label}")

# Show confidence scores as bar chart
st.subheader("Prediction Confidence")
fig2, ax2 = plt.subplots()
bars = ax2.bar(range(10), prediction[0] * 100, color='skyblue')
ax2.set_xticks(range(10))
ax2.set_xlabel('Digits')
ax2.set_ylabel('Confidence (%)')
ax2.set_ylim(0, 100)

# Highlight the predicted digit
bars[predicted_label].set_color('green' if predicted_label == actual_label else 'red')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom')

st.pyplot(fig2)


