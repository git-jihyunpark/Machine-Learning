# [Python] Machine Learning

🗓️ **Date**: 2024.03.04 ~ 2024.06.23

<br/>

📊 **Objective**
 1. Enhance proficiency in using `Python`.
 2. Develop predictive models using machine learning.
<br/>

🧩 **Table of Contents**
|Week|Content|
|----|-------|
|W01|[Linear Regression (1)](https://github.com/git-jihyunpark/Machine-Learning/blob/main/ML_W01_Linear%20regression.ipynb)|
|W02|[Linear Regression (2)](https://github.com/git-jihyunpark/Machine-Learning/blob/main/ML_W02_Linear%20regression2.ipynb)|
|W03|[Regression Tree (1)](https://github.com/git-jihyunpark/Machine-Learning/blob/main/ML_W03_Regression_tree.ipynb), [(2)](https://github.com/git-jihyunpark/Machine-Learning/blob/main/ML_W03_Regression_tree2.ipynb)|
|W04|[Classification Tree (1)](https://github.com/git-jihyunpark/Machine-Learning/blob/main/ML_W04_Classification%20tree.ipynb)|
|W05|[Classification Tree (2)](https://github.com/git-jihyunpark/Machine-Learning/blob/main/ML_W05_Classification%20tree2.ipynb)|
|W06|[Logistic Regression](https://github.com/git-jihyunpark/Machine-Learning/blob/main/ML_W06_Logistic%20Regression.ipynb)|
|W07|[Performance Measure](https://github.com/git-jihyunpark/Machine-Learning/blob/main/ML_W07_Performance_measure.ipynb)|
|W08|[RandomForest](https://github.com/git-jihyunpark/Machine-Learning/blob/main/ML_W08_Randomforest.ipynb)|
|W09|[Gradient Boosting (Regression)](https://github.com/git-jihyunpark/Machine-Learning/blob/main/ML_W09_Gradient%20Boosting(regression).ipynb)|
|W10|[Gradient Boosting (Classification)](https://github.com/git-jihyunpark/Machine-Learning/blob/main/ML_W10_Gradient%20Boosting(classification).ipynb)|
|W11|[Final Project](https://github.com/git-jihyunpark/Machine-Learning/blob/main/ML_Final.ipynb)|
<br/>


## 🔷 Final Project: Customer Churn Prediction


📌 **Introduction**
- Customer churn is a critical issue for businesses, and it is essential to support decision-making processes that minimize churn and promote customer retention.
- A predictive model is developed using supervised learning algorithms to identify potential customers who are likely to cancel their subscriptions to a hypothetical telecommunications service.
<br/>


📂 **Dataset**
- Churn Customer can be defined as a user who is likely to discontinue using the services. So, the target variable confirm if the customer has churned (1=yes; 0 = no).
- The data included 5.000 users and by the exploratory analysis, it is observed that:
    - 14% of the base are classified as churn.
    - 50% of the customers who called the company more than 3 times are classified as Churn.
    - 10% of those with no international plan are classified as Churn x 8% of those with an international plan are Churn.
<br/>
<br/>

### 1. Preprocessing

**1) Data Overview**
- Consists of 19 features and 1 target variable.
- A total of 5,000 records are included.

```python
df.info()
```
<img width="435" height="446" alt="image" src="https://github.com/user-attachments/assets/bafd94ce-8da5-432c-baa7-ca0fc73e20b9" /> <br/>




**2) Missing Value Check**
- No missing values detected.

```python
df.isnull().sum()
```
<img width="262" height="350" alt="image" src="https://github.com/user-attachments/assets/6bc2b5b4-cb86-4df7-8448-d99dfeddf008" /> <br/>


**3) Data Preprocessing**
- Performed one-hot encoding: converted yes/no values in the `international_plan` and `voice_mail_plan` features to 1/0.
```python
# 'international_plan', 'voice_mail_plan' 컬럼 정수형으로 변환 (Yes/No -> 1/0)
df_encoded = pd.get_dummies(df, columns=['international_plan', 'voice_mail_plan'], drop_first=True)

# astype() 메서드를 사용하여 boolean을 정수형으로 변환 (Yes/No -> 1/0)
df_encoded['international_plan'] = df_encoded['international_plan_yes'].astype(int)
df_encoded['voice_mail_plan'] = df_encoded['voice_mail_plan_yes'].astype(int)
df_encoded.drop(columns=['international_plan_yes', 'voice_mail_plan_yes'], inplace=True)
```
<br/>

- Removed unnecessary text: extracted only numeric values (e.g., ‘415’) from the `area_code` feature (e.g., area_code_415).
```python
# 'area_code' 컬럼 지역 번호만 추출
df_encoded['area_code'] = df_encoded['area_code'].str.split('_').str[-1]
```
<br/>

- Added a `total` feature: created by summing the `minutes`, `calls`, and `charges` features across different time periods.
```python
# 'total_minutes', 'total_calls', 'total_charge' 컬럼 추가
df["total_minutes"] = (df["total_day_minutes"] + df["total_eve_minutes"] + df["total_night_minutes"] + df["total_intl_minutes"])
df["total_calls"] = (df["total_day_calls"] + df["total_eve_calls"] + df["total_night_calls"] + df["total_intl_calls"])
df["total_charge"] = (df["total_day_charge"] + df["total_eve_charge"] + df["total_night_charge"] + df["total_intl_charge"])
```
<br/>
<br/>




### 2. EDA

**1) Comparison of Average Call Duration, Frequency, and Charges by Time Period**
- The `night` time period shows lower charges compared to the `day` and `eve` periods.
- There are no significant differences in call duration or call frequency across time periods.

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/7165c8e4-ddd6-4890-ae1a-3bdf35c5ca50" /> <br/><br/>


**2) Comparison of Average Call Duration, Frequency, and Charges by State**
- The highest average call duration is observed in `Kansas (612.18)`, while the lowest is in `Arizona (574.56)`.
<img width="1182" height="497" alt="image" src="https://github.com/user-attachments/assets/5f6209da-b5a7-4949-aa20-2f8013a19f1c" /> <br/><br/>

- The highest average number of calls is found in `North Dakota (314.31)`, while the lowest is in `South Dakota (296.08)`.
<img width="1182" height="497" alt="image" src="https://github.com/user-attachments/assets/3948432d-f0ef-4e22-9805-5de6bed2f8c5" /> <br/><br/>

- The highest average call charges are in `Kansas (62.30)`, whereas the lowest are in `Illinois (57.57)`.
<img width="1182" height="497" alt="image" src="https://github.com/user-attachments/assets/e3b31675-0dbf-45a3-9478-7d4d6f95a7f2" /> <br/><br/>


**3) Comparison of Churn Rate by Service Usage**
- Customers not using the international call service have a churn rate of 7%.
<img width="859" height="547" alt="image" src="https://github.com/user-attachments/assets/8fbdf46e-3e8f-4cd5-98b6-fc598da7023f" /> <br/><br/>

- Customers not using the voice mail service have a churn rate of 20%.
<img width="859" height="547" alt="image" src="https://github.com/user-attachments/assets/45060fb0-464c-409e-acc6-0d6ef409cf75" /> <br/><br/>


**4) Comparison of Churn Rate by Number of Customer Service Calls**
- Customers who contacted the customer service center once or twice have a churn rate of 12%.
<img width="859" height="547" alt="image" src="https://github.com/user-attachments/assets/aef28802-4e5c-4b77-91e8-fad725f46be0" /> <br/><br/>


**5) Relative Frequency Histogram**
- Most numerical variables exhibit a normal distribution, whereas `total_intl_calls` and `number_customer_service_calls` are skewed to the left.
<img width="1489" height="1490" alt="image" src="https://github.com/user-attachments/assets/76d21fb2-66d6-44b6-958d-699425a9075a" /> <br/><br/>


**6) Correlation Heatmap**
- There is a strong positive correlation between `minutes` and `charges`.
- The features `international_plan` and `number_customer_service_calls` show a relatively higher positive correlation with customer churn compared to other variables.
<img width="1111" height="988" alt="image" src="https://github.com/user-attachments/assets/042c2044-1982-4c67-b593-bc5974110e18" /> <br/>
<br/>
<br/>


### 3. Classification

**📍 Model Performance Comparison**
- Classification models were trained using `Decision Tree`, `Logistic Regression`, `Random Forest`, and `Gradient Boosting` algorithms.
- training:test = 80:20(random splitting)
- The `Random Forest` model demonstrated the highest predictive performance.
<br/>

|        |Decision Tree|Logistic Regression|Random Forest|Gradient Boosting|
|--------|-------------|-------------------|-------------|-----------------|
|Accuracy|0.893|0.850|0.941|0.883|
|F1-score|0.584|0.242|0.782|0.480|
|AUROC   |0.735|0.566|0.850|0.670|

<br/>
<br/>

**1) Decision Tree**
- Accuracy: 0.893, F1-score:0.584, AUC:0.735
<img width="896" height="509" alt="image" src="https://github.com/user-attachments/assets/e57ed198-b41f-4519-9191-af5cef82ae78" /> <br/><br/>


**2) Logistic Regeression**
- Accuracy: 0.850, F1-score:0.242, AUC:0.566
- Significant Coefficients; `international_plan`, `voice_mail_plan`, `total_intl_calls`, `number_customer_service_calls`
<img width="789" height="820" alt="image" src="https://github.com/user-attachments/assets/72f9ec9a-b4bb-4cf1-be53-e8bebf80c96b" /> <br/><br/>
<img width="767" height="575" alt="image" src="https://github.com/user-attachments/assets/c9414975-705a-4fb8-8249-860c3a702339" /> <br/><br/>

**3) Random Forest**
- Accuracy: 0.941, F1-score:0.782, AUC:0.850
- Feature Importance; `number_customer_service_calls`, `total_day_charge`, `total_day_minutes` <br/>
[RF_output](https://github.com/user-attachments/files/23409889/output.pdf)  <br/><br/>


**4) Gradient Boosting**
- Accuracy: 0.883, F1-score:0.480, AUC:0.670 <br/>
<img width="605" height="301" alt="image" src="https://github.com/user-attachments/assets/f459a5cb-9533-454d-96ea-79bbc6c97d36" /> <br/><br/>

<br/>
<br/>

---


💖 **Lesson & Learn**
1. Proficiency in using the `Python` Language
   > numpy, pandas, sklearn, seaborn, matplot
2. Understanding and Development Skills in Machine Learning 
   > Decision Tree, Logistic Regression, Random Forest, Gradient Boosting, Naive-bayse, k-NN 
3. Data Visualization Skills
   > Scatter plot, Heatmap, Histogram, Coverage-Homogeneity plot
