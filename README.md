# Early Detection of Clinical Bovine Mastitis using PySpark

Course project for **INT315: Cluster Computing**.

## About
Mastitis is an udder infection in cows that is often noticed only after visible symptoms appear. This project uses sensor readings (udder size, temperature) with cow details to predict whether a cow has mastitis, using **Apache Spark (PySpark)**. Spark SQL is used for exploration and Spark MLlib for the models.

Inspired by SIH26109. The original problem is hardware based, so this project covers the data and model side only.

## Dataset
- **Name:** Clinical Mastitis in Cows based on Udder Parameter using IoT
- **Source:** Mendeley Data - https://data.mendeley.com/datasets/kbvcdw5b4m/1
- **File:** `clinical_mastitis_cows_version1.csv`
- **Size:** 6,600 records, 1,100 cows, 6 days per cow, 18 columns
- **Target:** `class1` (0 = normal, 1 = mastitis)

## What the project does
1. Loads and explores the data with Spark SQL.
2. Checks for data leakage. `Milk_visibility` is identical to the label, and `Hardness` and `Pain` match it in over 99% of rows, so all three are dropped.
3. Splits the data 80/20 **by cow** so the same cow is never in both train and test.
4. Prepares features (StringIndexer, VectorAssembler, MinMaxScaler).
5. Trains Logistic Regression, Decision Tree, Naive Bayes and Linear SVM.
6. Evaluates with accuracy, precision, recall, F1, ROC-AUC and confusion matrices.

## Results (test set)
| Model | Accuracy | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.9838 | 0.8966 | 0.9455 |
| Decision Tree | 0.9829 | 0.8908 | 0.9422 |
| Naive Bayes | 0.8865 | 0.2759 | 0.4324 |
| Linear SVM | 0.9838 | 0.8966 | 0.9455 |

Recall and F1 are for the mastitis class. Precision was 1.0 for all models.

## How to run
1. Open `Mastitis_Detection_PySpark_Class_Level.ipynb` in Google Colab.
2. Run the cells in order.
3. When asked, upload `clinical_mastitis_cows_version1.csv`.

## Tools
Python 3, Apache Spark 4.0.4 (PySpark), Spark SQL, Spark MLlib, pandas, matplotlib, seaborn, Google Colab.

## Files
- `Mastitis_Detection_PySpark_Class_Level.ipynb` - notebook with the full code
- `Mastitis_Detection_Project_Report.pdf` - project report
- `clinical_mastitis_cows_version1.csv` - dataset (download from the link above)

## Limitations
Only 6 days of data per cow, so this is early detection and not long-term forecasting. Only 17 Holstein cows. A single train/test split was used with no tuning.
