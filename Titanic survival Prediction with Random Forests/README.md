# Titanic Survival Prediction with Random Forests
<p align="center">
<img src="../images/titanic.jpeg" width=400>

## Overview:
In this project, I implemented decision trees and random forests for classification on two datasets: 
1. the spam dataset. 
2. Titanic dataset to predict survivors of the infamous disaster.

Look at [Titanic survival Prediction with Random Forests.ipynb](https://github.com/JC01111/Machine-Learning-Projects/blob/main/Titanic%20survival%20Prediction%20with%20Random%20Forests/Titanic%20survival%20Prediction%20with%20Random%20Forests.ipynb) for my implementation. I first implemented _Decision Trees_, then I implemented a _Random Forest_ based on the decision trees I have.

Look at [Titanic survival Prediction with Random Forests_Sklearn.ipynb](https://github.com/JC01111/Machine-Learning-Projects/blob/main/Titanic%20survival%20Prediction%20with%20Random%20Forests/Titanic%20survival%20Prediction%20with%20Random%20Forests_Sklearn.ipynb) for the Scikit-learn version of decision trees and random forests methods for spam classification and Titanic prediction. 

___

These are the accuracies based on my implementation for `Spam` and `titanic` datasets:
```
Spam training accuracy for decisionTree: 0.8071111111111111,
Spam validation accuracy for decisionTree: 0.7708703374777975

Spam training accuracy for bagging: 0.8266666666666667,
Spam validation accuracy for bagging: 0.797291296625222

Spam training accuracy for Random Forest: 0.7431111111111111,
Spam validation accuracy for Random Forest: 0.7273534635879219

Titanic training accuracy for decisionTree: 0.8208955223880597,
Titanic validation accuracy for decisionTree: 0.7178217821782178

Titanic training accuracy for bagging: 0.845771144278607,
Titanic validation accuracy for bagging: 0.7896039603960396

Titanic training accuracy for Random Forest: 0.835820895522388,
Titanic validation accuracy for Random Forest: 0.780940594059406
```

Here is a partial decision trees split process on `Spam` dataset:
```
('exclamation') < 1.0
  ('parenthesis') < 1.0
    ('meter') < 1.0
      ('creative') < 1.0
        ('money') < 1.0
          ('pain') < 1.0
            ('ampersand') < 1.0
              ('dollar') < 1.0
               Predict: 0
Therefore this email was ham.

('money') >= 1.0
  ('business') < 1.0
    ('semicolon') < 2.0
      ('out') < 1.0
       Predict: 1
Therefore this email was spam.
```

Titanic Decision Trees Visualizatioin:
```
 ('female') < 1.0
   ('pclass') < 2.0
     ('age') < 17.0
      Predict: 1
     ('age') >= 17.0
      Predict: 0
   ('pclass') >= 2.0
     ('age') < 4.0
      Predict: 1
     ('age') >= 4.0
      Predict: 0
 ('female') >= 1.0
   ('pclass') < 3.0
     ('fare') < 31.6833
      Predict: 1
     ('fare') >= 31.6833
      Predict: 1
   ('pclass') >= 3.0
     ('fare') < 23.45
      Predict: 1
     ('fare') >= 23.45
      Predict: 0
```
Because the features of Titanic dataset are special, so the tree won't grow too deep (be careful of overfitting).
