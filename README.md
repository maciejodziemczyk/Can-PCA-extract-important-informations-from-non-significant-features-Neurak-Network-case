# Can PCA extract important informations from non-significant features - Neurak Network case
Project created for *Machine Learning 2: predictive models, deep learning, neural network* classes at WNE UW

Language:
 * English - classes, notebook, report, presentation, proposal

Semester: III (MA studies)

## About
The main objective of this project was to check if PCA applied to higlhy colinear data (bankruptcy prediction with financial indicators task) can improve overall Neural Network performance. This project was about modelling and learnt models application rather than data analysis. Nevertheless some preprocessing was carried out - decoding, balance and missings check. It turned out that there were many missings in this data, to handle it special data imputation algorithm was developed (which I'm very proud of and I decided to use it in my Master Thesis).

Data imputation algorithm description (brief):<br>
You can find that idea [here](https://www.youtube.com/watch?v=nyxTdL_4Q-Q&t=494s&ab_channel=StatQuestwithJoshStarmer). The algorithm is based on Random Forest, steps:
1. Attribute median is imputed
2. Random Forest classifier is built
3. Normalized proximity matrix is computed (distance measure based on RF node for the particular observations)
4. Dot product of proximity and current X matrix is computed (weighted average)
5. Repeat steps 2-5 several times or until convergence

Next, normalization (x-mu)/sigma was performed for PCA and neural nets. Next, k-Fold stratified Cross Validation experiments was performed for RF to find quasi-optimal hyperparameters (random search). 

To perform PCA in more sophisticated way it was needed to spot non-significant features. I used several methods: Random Forest feature importance score (thay's why I performed hyperparameter optimization before), Mutual Information, Spearman Rank Correlation (with target), Automatic General to Specyfic procedure based on logistic regression and Likelihood Ratio test (written from scratch), "Lasso" logistic regression betas vanishing while penalty increasing, Spearman Rank Correlation matrix (to find groups of correlated features). 

I trained XGBoost (random search in CV) as a benchmark for nets.

Due to data imbalance I decided to use AUC-PR and AUC-ROC as a metrics, tensorflow has different AUCs computation rules than scikit-learn, so I had to write my own training loop for nets (with early stopping on PR). To find appropriate architecture and hyperparameters I performed some experiments on single train/valid split (computation power saving) on 50 epochs. I assumed some basic architecture based on my prior knowledge. My first step was to inspect optimization algorithms, I tried many of them with different learning rates, momentum or batch size. It turned out that RMSprop was the best one. After that I started experimets with activation functions and I found that tanh on the first and on the sigmoid on second hidden layer worked the best.

Next I tried different regularization (L1 and L2) and dropout settings. Next step was to add third leyer, but it didn't help. The last step was to try different architectures (sets of hidden nodes combinations) and Batch Normalization testing.

My final model was 2 hidden layers with 60 units each, tanh and sigmoid activations respectively, 0.4 dropouts on both layers, no regularization and batch norm, 350 batch size and RMSprop with default settings as optimizer.

The next step was to perform n-Fold stratified Cross Validation and it turned out that 50 epochs is not enougn (no early stopping applied) and I stopped on 400, after that I gave the last chance for 3-layerd network and tested in CV (as another NN variant) and the results was quite surprising (a bit higher mertics for 40 unit 3-rd layer added with sigmoid activation, no regu and batch norm, 0.001 regularization). The NN's gets closer to the tree-based methods.

The next part was about PCA, firstly I inspected the whole training sets to check how varied the data is. After that I prepared special wrapper for NN training combined with PCA (it had to be performed in every split to prevent information leakage). My first run was the naive one - I applied PCA on the whole dataset to see what happen, and the results was slightly better than on the raw data. To perform more sophisticated PCA I analyzed my feature selection metrics and tried different groups to apply PCA on (for example I found featurex x,y,z non-significant and x,y correlated with each other so I named this as one group and apply PCA to explain some fixed variance ratio). 

After importance metrics analysis I tried different groupping rules and explained variance settings (expert analysis was the best, by simple RF_score < x was very close).

Next I performed CV for my nets with develpoed PCA, you can see the results below (note naive network is network without PCA)

![validation ROCs and PRs](https://github.com/maciejodziemczyk/Can-PCA-extract-important-informations-from-non-significant-features-Neurak-Network-case/blob/main/ML2results.png)

The last thing was to check statistical differences between naive and PCA models (wilcoxon median equality test for validation folds). It turned out that PCA statisticaly boosted 2-layer model performance in contrast to the 3-layer model. I tested if overfitting was lower when PCA applied (Null hypothesis accepted everytime => no statistical difference).

Findings:
 - PCA can extract important information from not imporant features in Neural Network case, but it is not a magic trick, the results are slighlty better. However PCA can be considered during predictive modelling of high dimensional data, especially when there are correlated features
 - Random Forest impurity based feature importance score gives the best results in terms of feature selection among considered ones: Mutual Information, Spearman correlation, General to Specyfic Logistic Regression LR test, L2 Logistic regression, but all of this methods are important and may be used successfully during analysis
 - If overfitting is not a problem, tree-based methods outperforms Neural Networks in current task. Extreme Gradient Boosting with gbtree or dart booster performance was the best, following by Random Forest (the easiest to implement)

In this project I gained more experience with Machine Learning models, especially with keras library, I learnt a lot about feature selection methods too. I gained a lot experience with Python programming for data science too. I practices my problem solving skills. It was a big project, but now I know some issues:
 1. No EDA and advanced preprocessing (during analysis for my Thesis I found monotonic transformations very useful (especially Box-Cox and Yeo-Johnson), outliers reduction via quantile clipping (winsorization) is also extremely useful)
 2. Data normalization on the whole dataset (potential information leakage)

Nevertheless I enjoyed this task and I like my solution, even with mentioned issues, I learnt a lot, and that is the key for me.

## Repository description
 - Can PCA extract important informations from non-significant features- Neural Network case..ipynb - Jupyter Notebook with my analysis (we had to defense our projects)
 - LassoResults.csv - results of lasso logistic regression feature selection
 - ML2_Maciej_Odziemczyk_project_presentation.pdf - project presentation (we had to present the project to the group)
 - ML2_Maciej_Odziemczyk_project_report.pdf - project report (part of assignment)
 - ML2_Project_proposal_M_Odziemczyk.pdf - project proposal (our ideas had to be accepted by teacher)
 - ML2results.png - graphs for github readme :)
 - XX_Random_Search.csv - random search results for particular algorithm
 - xx_.p - pickled results for models

## Technologies
 - Python (pandas, numpy, scipy, matplotlib.pyplot, scikit-learn, statsmodels, keras, xgboost)
 - Jupyter Notebook
 - LaTeX (report, presentation, proposal)

## Author
 - Maciej Odziemczyk


