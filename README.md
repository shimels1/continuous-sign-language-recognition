
# Continuous Sign Language Recognition

This project focuses on **Continuous Ethiopian Sign Language (EthSL) Recognition** using **Machine Learning models** such as **BiLSTM (Bidirectional Long Short-Term Memory) and CTC (Connectionist Temporal Classification)**. The system is designed to recognize continuous sign language gestures and convert them into text, aiding communication for the deaf and hard-of-hearing community.

## Screenshots


![ScreenShot](https://github.com/shimels1/continuous-sign-language-recognition/blob/main/screenshot/SLR_dashboard.PNG)

## Features

- **Real-time Sign Language Recognition**: Converts continuous Ethiopian Sign Language (EthSL) gestures into text.
- **Deep Learning Models**: Utilizes **BiLSTM** and **CTC** for sequence learning.
- **Dataset Preprocessing**: Processes video sequences for feature extraction.
- **Model Training & Evaluation**: Implements a deep learning pipeline for training and testing.
- **Scalable Architecture**: Can be extended to support other sign languages.

## Technologies Used

- **Python** (for deep learning & data processing)
- **TensorFlow / Keras** (for BiLSTM and CTC model training)
- **OpenCV** (for video frame extraction)
- **NumPy & Pandas** (for data handling)
- **Matplotlib & Seaborn** (for visualization)
- **Google Colab / Jupyter Notebook** (for experimentation)

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.8+
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/shimels1/continuous-sign-language-recognition.git
   ```

2. Navigate to the project directory:

   ```bash
   cd continuous-sign-language-recognition
   ```

3. Create a virtual environment:

   ```bash
   python -m venv env
   source env/bin/activate  # For Linux/Mac
   env\Scripts\activate     # For Windows
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

- The dataset should contain **video sequences** of Ethiopian Sign Language gestures.
- Use OpenCV to extract frames and preprocess the images for model training.
- Ensure proper **labeling and segmentation** for continuous recognition.

## Model Training

1. Preprocess the dataset:

   ```bash
   python preprocess_data.py
   ```

2. Train the BiLSTM-CTC model:

   ```bash
   python train_model.py
   ```

3. Evaluate the model:

   ```bash
   python evaluate_model.py
   ```

## Usage

- Run the real-time recognition system:

   ```bash
   python real_time_recognition.py
   ```

- The model will capture video input, process the gestures, and output the recognized text.


## Future Enhancements

- Improve model accuracy with **Transformer-based architectures**.
- Extend support for **more sign languages**.
- Optimize for **real-time performance** on edge devices.

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Shimels Alem

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```




