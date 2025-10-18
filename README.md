# 🎓 Structured ML Pipeline for Student Performance Prediction

## 🧠 Project Overview
This project aims to **predict student performance (marks)** using a **supervised learning regression model** built with a fully structured and modular ML pipeline.  
The focus of the project is not just accuracy — but **clean architecture, reproducibility, scalability, and deployment-readiness**.  

It follows a component-based structure with stages for:
- Data ingestion
- Preprocessing
- Feature engineering
- Model training & tuning
- Model evaluation
- Deployment using Flask + WSGI

---

## 🚀 Key Features
- **End-to-End ML Pipeline**: Modular components with clean separation of concerns.  
- **Automated Data Preprocessing**: Handles missing values, feature scaling, and encoding.  
- **Hyperparameter Tuning**: Compared multiple regression models using GridSearchCV.  
- **Model Selection**: Chose the best-performing model based on R² and RMSE scores.  
- **Deployment Ready**: Flask + WSGI integration for scalable web serving.  
- **Maintainable Architecture**: Easy to extend for future datasets or regression problems.

---

## 🧩 Tech Stack
- **Programming Language:** Python  
- **ML Libraries:** Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn  
- **Deployment:** Flask, WSGI  
- **Development Tools:** Git, Jupyter Notebook  

---

## 🧪 Model Performance
| Metric | Score |
|--------|--------|
| **R² Score** | **8.88** |
| **Best Model** | RandomForestRegressor (tuned) |

---

## ⚙️ Project Structure
student-performance-prediction/
│
├── data/
│ └── student_scores.csv
│
├── src/
│ ├── components/
│ │ ├── data_ingestion.py
│ │ ├── data_preprocessing.py
│ │ ├── feature_engineering.py
│ │ └── model_trainer.py
│ │
│ ├── utils/
│ │ └── helpers.py
│ │
│ └── pipeline/
│ └── training_pipeline.py
│
├── app.py # Flask application
├── requirements.txt
├── README.md
└── model.pkl # Trained model artifact


---

## 🧮 How It Works
1. **Data Ingestion:** Load student dataset and split into training/testing sets.  
2. **Preprocessing:** Handle null values, scale features, and encode categorical variables.  
3. **Feature Engineering:** Derive meaningful features to improve model performance.  
4. **Model Training:** Train multiple regression models (Linear, Random Forest, etc.).  
5. **Evaluation:** Compare models and choose the best one using R² score.  
6. **Deployment:** Expose prediction endpoint via Flask + WSGI.

---

## ▶️ Running the Project
```bash
# Clone the repository
git clone https://github.com/Ayaan-Skh/student-performance-prediction.git
cd student-performance-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py

POST /predict
{
  "study_hours": 6,
  "attendance": 0.9,
  "assignments_submitted": 12
}

