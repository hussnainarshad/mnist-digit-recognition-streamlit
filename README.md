🧠 mnist-digit-recognition-streamlit
This project is an interactive web application that showcases digit recognition using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Users can explore test samples, visualize input images, and view the model's prediction confidence in real-time.
🚀 Features

Trained CNN Model: Built with Keras for accurate digit recognition.
Interactive Interface: Streamlit-powered viewer for seamless user interaction.
Real-Time Predictions: Displays model predictions on MNIST test images.
Confidence Visualization: Bar chart showing prediction probabilities for each digit.
Flexible Sample Selection: Choose samples manually or use a random sample button.

📁 Project Structure
mnist-digit-recognition-streamlit/
├── app.py                  # Streamlit application script
├── train_model.py          # Script to train the CNN model
├── model/
│   └── mnist_model.keras   # Pre-trained Keras model
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

🧪 Model Details

Dataset: MNIST (28x28 grayscale images of handwritten digits)
Architecture:
Conv2D (32 filters) → Conv2D (64 filters) → MaxPooling → Dropout (0.25)
Flatten → Dense (256 units) → Dropout (0.5) → Dense (10 units, softmax)


Loss Function: Categorical Crossentropy
Optimizer: Adam
Performance: ~99% accuracy on training set, ~98% on test set

💻 Getting Started
Prerequisites

Python 3.11+
Git

Installation

Clone the Repository
git clone https://github.com/hussnainarshad/mnist-digit-recognition-streamlit.git
cd mnist-digit-recognition-streamlit


Install Dependencies
pip install -r requirements.txt


Train the Model (Optional, pre-trained model included)
python train_model.py


Launch the Streamlit App
streamlit run app.py


Open your browser and navigate to http://localhost:8501 to use the app.


📝 Notes

Ensure all dependencies are installed correctly to avoid runtime errors.
The pre-trained model (mnist_model.keras) is included, so training is optional.
For custom modifications, edit train_model.py to adjust the model architecture or hyperparameters.

🙌 Contributing
Feel free to fork the repository, make improvements, and submit pull requests. Suggestions for new features or bug fixes are welcome!
