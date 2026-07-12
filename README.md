# Student Performance Indicator

This project explores how a student's exam performance (math, reading, and writing scores) is influenced by demographic and social factors such as gender, ethnicity, parental education, lunch type, and test preparation course completion. It follows the complete lifecycle of a machine learning project — from problem understanding to model selection.

## 📁 Repository Structure

```
Student-Performance-Indicator/
│
├── 1. EDA STUDENT PERFORMANCE.ipynb   # Exploratory Data Analysis notebook
├── 2. MODEL TRAINING.ipynb            # Model training and evaluation notebook
├── stud.csv                           # Dataset
└── LICENSE
```

## 🔄 Life Cycle of the ML Project

1. Understanding the Problem Statement
2. Data Collection
3. Data Checks to Perform
4. Exploratory Data Analysis (EDA)
5. Data Pre-Processing
6. Model Training
7. Choose Best Model

## 1) Problem Statement

This project understands how a student's performance (test scores) is affected by other variables such as **Gender, Ethnicity, Parental level of education, Lunch,** and **Test preparation course**.

## 2) Data Collection

- **Dataset Source:** [Kaggle – Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977)
- The data consists of **8 columns** and **1000 rows**.

### 2.2 Dataset Information

| Column | Description |
|---|---|
| `gender` | Sex of student (Male/Female) |
| `race_ethnicity` | Ethnicity of student (Group A, B, C, D, E) |
| `parental_level_of_education` | Parents' final education (bachelor's degree, some college, master's degree, associate's degree, high school) |
| `lunch` | Type of lunch before test (standard or free/reduced) |
| `test_preparation_course` | Completed or not completed before test |
| `math_score` | Math test score |
| `reading_score` | Reading test score |
| `writing_score` | Writing test score |

## 3) Data Checks Performed

- Check missing values
- Check duplicates
- Check data types
- Check number of unique values per column
- Check statistical summary of the dataset
- Check categories present in each categorical column

The dataset was found to be clean, with **no missing values** and no significant data quality issues.

## 4) Exploratory Data Analysis (EDA)

### 4.1 Score Distribution
Visualized the distribution of the average score using **Histogram** and **Kernel Density Estimation (KDE)** plots, both overall and split by gender.

### 4.4 Feature-Wise Visualization

**Gender**
- Balanced dataset: **Female students – 518 (52%)**, **Male students – 482 (48%)**

**Key Insights from EDA:**
- Number of male and female students is almost equal.
- Number of students is greatest in **Group C**.
- Students with **standard lunch** outperform those with free/reduced lunch.
- Most students have **not enrolled** in the test preparation course.
- Parental education level **"Some College"** is the most common, closely followed by **"Associate's Degree"**.

### 4.4.6 Outlier Detection
Boxplots were used to check for outliers across `math_score`, `reading_score`, `writing_score`, and `average` score.

### 4.4.7 Multivariate Analysis (Pairplot)
A pairplot (hue = gender) revealed that **all three scores (math, reading, writing) increase linearly with each other**, indicating strong positive correlation between subjects.

## 5) EDA Conclusions

- Student performance is related to **lunch type, race/ethnicity, and parental level of education**.
- **Females lead** in pass percentage and are also the top scorers.
- Student performance is **not strongly related** to the test preparation course on its own.
- However, **finishing the preparation course is beneficial** to score improvement.

## 6) Model Training

### 6.1 Data Preparation
- Target variable: `math_score`
- Features: `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`, `reading_score`, `writing_score`
- Categorical variables were encoded and features were preprocessed before training.

### 6.2 Models Trained
Multiple regression models were trained and compared using `RandomizedSearchCV` for hyperparameter tuning:

- Linear Regression
- Ridge Regression
- Lasso Regression
- K-Neighbors Regressor
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- CatBoost Regressor
- AdaBoost Regressor

### 6.3 Evaluation Metrics
Each model was evaluated on train and test sets using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score**

```python
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square
```

## 📊 Results

| Model Name | R² Score |
|---|---|
| **Ridge** | **0.8806** |
| Linear Regression | 0.8803 |
| CatBoosting Regressor | 0.8516 |
| AdaBoost Regressor | 0.8498 |
| Random Forest Regressor | 0.8473 |
| Lasso | 0.8253 |
| XGBRegressor | 0.8216 |
| K-Neighbors Regressor | 0.7838 |
| Decision Tree | 0.7603 |

**Best Model: Ridge Regression**, with an R² Score of **~0.88**, closely followed by Linear Regression. The Linear Regression model achieved a test accuracy of **88.03%**.

```python
lin_model = LinearRegression(fit_intercept=True)
lin_model = lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test)
score = r2_score(y_test, y_pred) * 100
print("Accuracy of the model is %.2f" % score)
# Accuracy of the model is 88.03
```

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/arpitsharma73/Student-Performance-Indicator.git
   cd Student-Performance-Indicator
   ```
2. Install dependencies (pandas, numpy, matplotlib, seaborn, scikit-learn, catboost, xgboost):
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn catboost xgboost
   ```
3. Run the notebooks in order:
   - `1. EDA STUDENT PERFORMANCE.ipynb`
   - `2. MODEL TRAINING.ipynb`

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, CatBoost

## 📜 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## 🙋 Author

**Arpit Sharma** — [GitHub Profile](https://github.com/arpitsharma73)
