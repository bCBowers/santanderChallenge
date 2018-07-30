This project is a solution to the Santander Value Prediction Challenge: https://www.kaggle.com/c/santander-value-prediction-challenge. I joined this challenge to expirement with ensemble models like LightGBM and XGBoost. My solution combines an approaching using Light Gradient Boosting and Extreme Gradient Boosting with an approach using Category Boosting Rgression. As of 7/10, I'm in 649 place out of 2789 teams and I'm currently working alone.

My model combines:

- LightGBM and Extreme Gradient Boosting

- Feature Reduction/Clustering and Category Boosting Regression

The individual models were developed with the help of kernels from Kaggle.

File Descriptions:

santanderv1.py -> ensemble model code

exploreSantander.ipynb -> contains exploratory data analysis and plots with Santander data set

submission.csv -> output code for submission to Kaggle

sample_submission.csv -> file provided by the Kaggle challenge

train.csv, test.csv -> these files are too big for Github but can be found on Kaggle: https://www.kaggle.com/c/santander-value-prediction-challenge/data
