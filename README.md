This project is a solution to the Santander Value Prediction Challenge: https://www.kaggle.com/c/santander-value-prediction-challenge. I joined this challenge to expirement with ensemble models like LightGBM and XGBoost. My solution combines four different ensemble approaches using a simple averages. The challenge has only been live for a few days, so I plan to iterate and improve on my solution. As of 6/26, I'm in 392 place out of 1389 teams and I'm currently working alone.

The four models applied so far are:

- Feature Reduction and LightGBM

- LightGBM

- Feature Reduction/Clustering and Category Boosting Regression

- Extreme Gradient Boosting

As expected, an average of the four techniques produces better results than any technique individually.The individual models were developed with the help of kernels from Kaggle.

File Descriptions:

santanderv1.py -> ensemble model code

exploreSantander.ipynb -> contains exploratory data analysis and plots with Santander data set

submission.csv -> output code for submission to Kaggle

sample_submission.csv -> file provided by the Kaggle challenge

train.py, test.py -> these files are too big for Github but can be found on Kaggle: https://www.kaggle.com/c/santander-value-prediction-challenge/data
