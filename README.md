# 🛳 Titanic Survival Exploratory Data Analysis (EDA)

This project performs an in-depth Exploratory Data Analysis (EDA) on the Titanic dataset, one of the most famous datasets used in machine learning and data science. The goal is to uncover meaningful patterns, trends, and insights into what factors contributed to passenger survival.

---

## 📁 Dataset

The dataset used is the **Titanic dataset** from [Kaggle](https://www.kaggle.com/competitions/titanic) or [OpenML](https://www.openml.org/d/40945). It contains information about passengers such as:

- PassengerID
- Name, Age, Sex
- Pclass (Ticket Class)
- Fare
- Embarked (Port of Embarkation)
- SibSp (Siblings/Spouses aboard)
- Parch (Parents/Children aboard)
- Survived (Target variable)

---

## 🎯 Objectives

- Understand the structure and content of the Titanic dataset.
- Perform data cleaning and preprocessing.
- Analyze relationships between features and survival rate.
- Visualize key patterns using plots and charts.
- Derive meaningful insights to guide future model building.

---

## 🔍 Key Steps

1. **Data Loading**
   - Load CSV into a Pandas DataFrame.
   - Check shape, types, and basic stats.

2. **Data Cleaning**
   - Handle missing values (e.g. Age, Embarked).
   - Drop or fill irrelevant/missing features.
   - Convert categorical variables.

3. **Exploratory Analysis**
   - Survival by gender, class, age group.
   - Correlation heatmaps.
   - Histograms, bar plots, and boxplots.
   - Grouped survival rate comparisons.

4. **Feature Engineering (Optional)**
   - Create age groups.
   - Create family size.
   - Encode categorical features.

---

## 📊 Sample Visualizations

- Bar plot: Survival rate by Gender
- Pie chart: Embarked locations
- Heatmap: Correlation between numerical features
- Boxplot: Age vs Survival
- Countplot: Pclass vs Survival

---

## 💡 Insights

- Females had a much higher survival rate than males.
- Passengers in 1st class were more likely to survive.
- Children under 10 had higher survival rates.
- Embarkation port 'C' had a higher percentage of survivors.
- Family size played a role in survival likelihood.

---

## 🛠 Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 📂 File Structure
Titanic-EDA/
├── Titanic.ipynb # Main Jupyter notebook with EDA code
├── train.csv # Dataset used for analysis
├── README.md # Project documentation



---

## 📌 To Run

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/titanic-eda.git
   cd titanic-eda


