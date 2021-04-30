# Can PCA extract important informations from non-significant features - Neurak Network case
Project created for *Machine Learning 2: predictive models, deep learning, neural network* classes at WNE UW

Language:
 * English - classes, notebook, report, presentation

Semester: III (MA studies)

## About
The main objective of this project was to check if PCA applied to higlhy colinear data (bankruptcy prediction with financial indicators task) can improve overall Neural Network performance. This project was about modelling and learnt models application rather than data analysis. Nevertheless some preprocessing was carried out - decoding, balance and missings check. It turned out that there were many missings in this data, to handle it special data imputation algorithm was developed (which I'm very proud of and I decided to use it in my Master Thesis).

Data imputation algorithm description (brief):<br>
You can find that idea [here](https://www.youtube.com/watch?v=nyxTdL_4Q-Q&t=494s&ab_channel=StatQuestwithJoshStarmer). The algorithm is based on Random Forest, steps:
1. Attribute median is imputed
2. Random Forest classifier is built
3. Normalized proximity matrix is computed (distance measure based on RF node for particular observations)
4. Dot product of proximity and current values is computed (weighted average)
5. Repeat steps 2-5 several times or until convergence

Next normalization (x-mu)/sigma was performed for future PCA and neural nets. Next k-Fold stratified Cross Validation experiments was performed for RF to find quasi-optimal hyperparameters (random search). 

To perform PCA in more sophisticated way it was needed to spot non-significant features I wanted to use several methods: Random Forest feature importance score (thay's why I performed hyperparameter optimization before), Mutual Information, Spearman Rank Correlation (with target), Automatic General to Specyfic procedure based on logistic regression and Likelihood Ratio test (written from scratch), "Lasso" logistic regression betas vanishing while penalty increasing, Spearman Rank Correlation matrix.
