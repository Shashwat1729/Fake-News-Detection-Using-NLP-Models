# Fake News Detection Using NLP Models

This project uses **Natural Language Processing (NLP)** to detect whether a news article is **real** or **fake**. The model is trained using a dataset of real and fake news, applying **TF-IDF** for feature extraction and **Logistic Regression** for classification.

## Features:
- Paste a news article into the web interface to predict whether it is real or fake.
- **Logistic Regression** is used as the classification algorithm.
- The project uses **TF-IDF** to convert news articles into numerical features.

## Dataset:
- The dataset used in this project consists of real and fake news labeled accordingly.
- You can download a suitable dataset from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) and place it in the `data/` folder.

## Setup:

1. Clone this repository:
   ```bash
   git clone https://github.com/Shashwat1729/Fake-News-Detection-NLP.git
   cd Fake-News-Detection-NLP
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download a fake news dataset and place it in the `data/` folder:
   ```bash
   mkdir data
   # Download the dataset from Kaggle and place it in the folder
   ```

4. Train the model (optional if you don’t want to use the pre-trained model):
   ```bash
   python model.py
   ```

5. Run the Flask app:
   ```bash
   python app.py
   ```

6. Open your browser and navigate to `http://127.0.0.1:5000/` to use the fake news detector.

**Contributions**:
Feel free to contribute by opening issues or creating pull requests!
