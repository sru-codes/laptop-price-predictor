# Laptop Price Predictor (Minor Project)

**Authors:** Srustisri Panda, Puja Rani Mishra, Kajal Roul  
**Project Type:** Group Project  
**Topic:** Regression & Prediction Projects  
**Project ID:** Minor Project #6 (from AI/ML Project List)

---

## 📌 Project Overview
The **Laptop Price Predictor** is a machine learning application designed to estimate the market price of laptops based on various hardware specifications. This project demonstrates the application of **Linear Regression** in a real-world scenario where multiple factors influence the final cost of a product.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, NumPy, Scikit-learn, Joblib
- **Algorithm:** Linear Regression

## 📊 Features & Functionality
- **Data Preprocessing:** Handles categorical variables like 'Brand' using One-Hot Encoding.
- **Predictive Modeling:** Utilizes a Linear Regression model to find relationships between RAM, Storage, Processor Speed, and Price.
- **Model Persistence:** Saves the trained model and feature list for future use without retraining.
- **Evaluation:** Measures performance using Mean Squared Error (MSE) and R-squared metrics.

## 🚀 Getting Started

### Prerequisites
Ensure you have Python installed. You can install the required libraries using:
```bash
pip install pandas scikit-learn joblib
```

### Running the Project
1. **Train the Model:**
   ```bash
   python train.py
   ```
2. **Predict:**
   The training script will output the model's accuracy and save a `.pkl` file which can be integrated into a web or desktop application.

---
*Developed as part of the AI/ML with Python Course.*
