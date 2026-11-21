# Garment Worker Productivity Prediction

## Project Overview
This project analyzes a garment manufacturing worker productivity dataset and builds machine learning regression models to predict **actual productivity**. The notebook covers dataset inspection, data cleaning, outlier analysis, exploratory data analysis (EDA), feature engineering, preprocessing, model training, and evaluation.

## Objective
To explore the factors recorded in the garment manufacturing process and use machine learning regression algorithms to predict the `actual_productivity` value.

## Dataset
- **CSV file expected by the notebook:** `garments worker productivity.csv`
- **Target variable:** `actual_productivity`
- **Dataset:** Garment worker productivity records, including production, department, team, incentive, worker-count, and other recorded attributes.

Keep the CSV file in the same folder as the Jupyter Notebook. If the filename is different, update the filename in the `pd.read_csv()` statement.

## Libraries and Tools
- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow

### 1. Importing Libraries
Imports the libraries required for numerical calculations, data manipulation, visualization, and machine learning.

### 2. Loading the Dataset
Loads the CSV file using Pandas and stores it in a DataFrame named `df`.

### 3. Dataset Inspection
The notebook uses:
- `head()` to display the first five records.
- `tail()` to display the last five records.
- `info()` to inspect column data types and non-null values.
- `shape` to view the number of rows and columns.
- `columns` to list the column names.
- `describe()` to generate descriptive statistics.

### 4. Data Cleaning
- Checks missing values in each column.
- Checks and removes duplicate rows.
- Converts the `date` column to datetime format.
- Removes rows where `actual_productivity` is missing.
- Fills missing numerical feature values with their median.
- Fills missing categorical values with their mode.
- Displays the missing-value count after handling.

### 5. Outlier Detection and Removal
The notebook uses the **Interquartile Range (IQR)** method on numerical columns.

The IQR is calculated as:

`IQR = Q3 - Q1`

The lower and upper limits are:

- `Lower bound = Q1 - 1.5 × IQR`
- `Upper bound = Q3 + 1.5 × IQR`

A row is included in `df_clean` only if none of its numerical values fall outside these limits. The notebook displays boxplots before and after filtering and prints the dataset shapes before and after removal.

**Important implementation note:** The notebook creates the filtered DataFrame as `df_clean`, but the later EDA, feature engineering, train-test split, and model training cells use `df`, not `df_clean`. Therefore, the current notebook's model results are based on `df` and do not apply the outlier-filtered dataset. To train on the filtered data, assign `df = df_clean.copy()` after reviewing the outlier-removal results and before the later analysis cells.

Outliers are not necessarily errors; inspect them before removing them.

### 6. Exploratory Data Analysis (EDA)
The notebook visualizes and investigates:
- Distribution of actual worker productivity.
- Actual productivity by department.
- Average productivity by day.
- Number of workers compared with actual productivity, with department shown as a grouping variable.
- Incentive compared with actual productivity.
- Correlation between numerical variables using a heatmap.

### 7. Feature Engineering
Extracts the following date-related features:
- `month`
- `day_of_month`
- `day_of_week`

The original `date` column is then removed.

### 8. Independent and Dependent Variables
- **Independent variables (`X`):** All columns except `actual_productivity`.
- **Dependent variable (`y`):** `actual_productivity`.

### 9. Train-Test Split
Splits the data into:
- 80% training data.
- 20% testing data.

The split uses `random_state=42` to make the split reproducible.

### 10. Data Preprocessing
A Scikit-learn `ColumnTransformer` and pipelines are used:
- **Numerical features:** Missing-value imputation using the median, followed by standard scaling.
- **Categorical features:** Missing-value imputation using the most frequent value, followed by one-hot encoding.

The preprocessing steps are fitted as part of each model pipeline.

### 11. Machine Learning Algorithms
The notebook trains three regression models:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

The Random Forest model uses 200 estimators.

### 12. Model Evaluation
The models are evaluated using:
- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted values.
- **MSE (Mean Squared Error):** Average squared prediction error.
- **RMSE (Root Mean Squared Error):** Square root of MSE.
- **R² Score:** Measures the proportion of target variation explained by the model.

The results are sorted by RMSE, and the model with the lowest RMSE is selected as the best model in this notebook's comparison.

### 13. Model Comparison and Predictions
The notebook:
- Compares model RMSE scores using a bar chart.
- Compares model R² scores using a bar chart.
- Plots actual productivity against predicted productivity for the selected model.
- Displays the first ten actual and predicted productivity values.

## How to Run the Project

1. Install Python and Jupyter Notebook.
2. Install the required libraries:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

3. Place the notebook and `garments worker productivity.csv` in the same folder.
4. Open Jupyter Notebook.
5. Open the project notebook.
6. Run the cells in order, from top to bottom.

## Conclusion
This project demonstrates the use of exploratory data analysis and machine learning for garment worker productivity prediction. It examines the dataset, handles missing and duplicate records, identifies potential outliers, visualizes relationships in the data, prepares features, trains regression models, and compares their predictive performance using standard evaluation metrics.

The best-performing algorithm is determined from the actual evaluation results generated when the notebook is executed.
