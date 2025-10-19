# ANN-Classification-Churn

A machine learning project designed to predict customer churn using an **Artificial Neural Network (ANN)** model. Built with Python and deployed using **Streamlit**, this project demonstrates how deep learning can be applied to classification problems with structured data.

**Live App:** [Click here to open the deployed app](https://ann-classification-churn-sdzp23npe44dffcwvqkvbe.streamlit.app/)

***

## 📌 Project Overview

Customer churn prediction helps businesses identify customers likely to discontinue their services. By forecasting churn, companies can proactively improve customer retention strategies.

This project builds, trains, and evaluates an ANN model to predict churn probabilities based on customer behavior and demographic features.

***

## ⚙️ Tech Stack

- **Language:** Python
- **Frameworks/Libraries:** TensorFlow, Keras, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn
- **Deployment:** Streamlit
- **Tools:** Jupyter Notebook

***

## 🧠 Model Architecture

- Input Layer – Encodes customer features after preprocessing
- Hidden Layers – Two dense layers with ReLU activation
- Output Layer – Single neuron with sigmoid activation for binary classification

```
Input → Dense (ReLU) → Dense (ReLU) → Output (Sigmoid)
```

- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Metric:** Accuracy

***

## 🧩 Dataset

The dataset contains customer demographic and activity data including features such as:

- Credit score
- Geography (One-Hot Encoded)
- Gender
- Age
- Tenure
- Balance
- Salary
- Number of products used

Target variable: **Exited** (1 = churned, 0 = retained)

***

## 🚀 Workflow

1. **Data Preprocessing**
    - Handled missing values
    - Applied One-Hot Encoding to categorical features
    - Normalized numerical data using StandardScaler
2. **Model Development**
    - Built and compiled an ANN model with Keras Sequential API
    - Trained with an 80/20 train-test split
3. **Model Evaluation**
    - Used accuracy and confusion matrix for performance assessment
4. **Deployment**
    - Created a Streamlit user interface for prediction
    - Integrated trained model for live inference

***

## 📊 Sample Output

Output includes:

- Churn probability
- Prediction label (Churn / Not Churn)

The Streamlit dashboard allows real-time user input for dynamic predictions.

***

## 🧾 Results

- Model achieved above **80% accuracy** on validation data.
- Optimized using batch normalization and dropout for stability.

***

## 🛠️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/saurav3k2/ANN-Classification-Churn.git
cd ANN-Classification-Churn

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```


***

## 📈 Future Improvements

- Experiment with feature importance using SHAP or LIME
- Implement hyperparameter tuning with GridSearchCV or Keras Tuner
- Deploy scalable version on cloud (AWS / Heroku)

***

## 👨‍💻 Author

**Saurav Kumar**
Data Science Enthusiast | Machine Learning Developer
[GitHub Profile](https://github.com/saurav3k2)

***

Would you like a **README variant including project screenshots** or badges (like model accuracy, Python version)? I can add those next for a more polished GitHub presentation.
<span style="display:none">[^1]</span>

<div align="center">⁂</div>

[^1]: https://github.com/saurav3k2/ANN-Classification-Churn

