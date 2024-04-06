questions = [

    #### Supervised Learning Section

    # Fundamentals of Supervised Learning (15)
    "How does supervised learning differ from unsupervised and reinforcement learning in the context of training objectives and data requirements?",
    "What are the key components of a supervised learning model's training data, and why is the quality of this data crucial for model performance?",
    "How do you decide which supervised learning algorithm is most appropriate for a particular problem statement or dataset?",
    "What metrics are commonly used to evaluate the performance of supervised learning models, and how do they differ for regression versus classification tasks?",
    "What strategies can be employed to prevent overfitting in supervised learning models?",
    "How does the concept of bias-variance tradeoff apply to supervised learning, and what implications does it have for model complexity and generalization?",
    "Discuss the role of gradient descent in the optimization of supervised learning models. How do variations like stochastic gradient descent (SGD) and mini-batch gradient descent contribute to model training?",
    "What are some common challenges faced when scaling supervised learning models to large datasets, and how can these be mitigated?",
    "Explain the concept of feature engineering and selection in supervised learning. How do they impact model performance and interpretability?",
    "Why is data preprocessing important in supervised learning?",
    "How do you handle missing data in a supervised learning model?",
    "What are the two main types of tasks in supervised learning, and how are they different??"
    "Explain the concept of early stopping in training models.",
    "Describe the process of cross-validation in supervised learning.",
    "What are some common metrics used to evaluate the performance of classification models in supervised learning?",

    # SWM (10)
     "What is the principle behind Support Vector Machines (SVM), and how do they perform classification tasks?",
    "Explain the concept of a hyperplane in the context of SVM. How does SVM determine the optimal hyperplane?",
    "What are margins in SVM, and why are they important?",
    "Define support vectors and discuss their role in SVM. How does changing the support vectors affect the model?",
    "What is the kernel trick, and how does it allow SVM to perform well with non-linearly separable data?",
    "Explain the concept of soft margin in SVM. How does it help in handling misclassified samples and outliers?",
    "How does the regularization parameter C in SVM influence the trade-off between achieving a low training error and maintaining a large margin?",
    "Discuss the role of the epsilon parameter in SVR. How does it influence the model's sensitivity to training data?",
    "What are the computational challenges associated with training SVM models, especially with large datasets?"
    "Give examples of real-world applications where SVM models are effectively used. Why is SVM a preferred choice in these scenarios?"
    "Discuss potential ethical implications of using SVM in sensitive applications. How can bias in the training data affect SVM predictions?",
    "What steps can be taken to ensure that SVM models are fair and unbiased?",

    # decision trees
    "What is a decision tree, and how does it make predictions?",
    "Describe how a decision tree is built. What does splitting mean in the context of decision trees?",
    "Define and explain the importance of root node, branches, and leaf nodes in a decision tree.",
    "Discuss strategies to prevent overfitting in decision trees, such as pruning and setting minimum split criteria.",
    "What are the advantages of using decision trees over other machine learning algorithms?",
    "Discuss the limitations of decision trees. How can these limitations affect their performance and applicability to certain types of problems?",
    "Describe how decision trees can be used in multi-class classification problems. Are there any specific considerations or adjustments needed?",
    "How do you evaluate the performance of a decision tree model? Discuss the metrics and methods used for evaluation.",
    "What are the key hyperparameters in decision tree models, and how do they impact the model's complexity and performance?",
    "Compare the use of decision trees in Random Forests versus Gradient Boosted Decision Trees (GBDT). How do their strategies and outcomes differ?",
    "How can bias in the training data or decision-making criteria of a decision tree lead to unfair or biased outcomes?",
    "What are potential ethical concerns when using decision trees in decision-making processes, especially in sensitive areas like healthcare or criminal justice?",

    # Logistic regression
    "What is logistic regression, and how does it differ from linear regression?",
    "Explain the sigmoid function and its importance in logistic regression.",
    "How is the logistic regression model formulated, and what does the logistic function represent in this context?",
    "Describe the Maximum Likelihood Estimation (MLE) method in the context of logistic regression. How is it used to estimate the model parameters?",
    "How does logistic regression perform binary classification? What modifications are made to apply it to multiclass classification scenarios?",
    "Discuss the concepts of One-vs-Rest (OvR) and One-vs-One (OvO) strategies in multiclass classification with logistic regression.",
    "What metrics are commonly used to evaluate the performance of a logistic regression model in classification tasks?",
    "Explain the concept of the confusion matrix and how it applies to logistic regression model evaluation.",
    "What is regularization, and why is it important in logistic regression?",
    "How does feature selection impact the performance of logistic regression models?",
    "Explain how logistic regression can be extended or modified to handle imbalanced datasets.",
    "Discuss the limitations of logistic regression and potential strategies to overcome these limitations.",
    "Provide examples of real-world applications where logistic regression is an effective modeling technique.",
    "How can logistic regression be used in predictive analytics for customer churn, fraud detection, or disease diagnosis?",
    "How can the outcomes of a logistic regression model reflect bias present in the training data, and what steps can be taken to mitigate this issue?",

    # Naive bayes 
    "What is the Naive Bayes theorem, and how is it used in classification tasks?",
    "Explain the 'naive' assumption in Naive Bayes classifiers. Why is it considered naive?",
    "Describe the different types of Naive Bayes classifiers and the scenarios in which each type is most appropriate.",
    "Compare and contrast Gaussian, Multinomial, and Bernoulli Naive Bayes classifiers.",
    "How is a Naive Bayes classifier trained on a dataset? What does the training process involve?",
    "Explain how a Naive Bayes classifier makes predictions for new, unseen data.",
    "Discuss the implications of the independence assumption for feature selection in Naive Bayes models. How realistic is this assumption in real-world applications?",
    "How does the independence assumption affect the classifier's performance and accuracy?",
    "What are the advantages of using Naive Bayes classifiers over other classification algorithms?",
    "Discuss the limitations of Naive Bayes classifiers. Under what circumstances might they perform poorly?",
    "How do you evaluate the performance of a Naive Bayes classifier? Discuss the metrics and methods typically used.",
    "Provide examples of real-world applications where Naive Bayes classifiers are effectively used. Why are they a preferred choice in these scenarios?",
    "Discuss how Naive Bayes classifiers can be used in text classification tasks such as spam detection and sentiment analysis.",
    "What strategies can be employed to improve the performance of a Naive Bayes classifier?",
    "What are some ethical considerations to keep in mind when using Naive Bayes classifiers in sensitive applications, such as predictive policing or hiring practices?",
    "Discuss how bias in training data can affect Naive Bayes classifiers and steps that can be taken to mitigate such biases.",

    # polynomial regression
    "What is polynomial regression, and how does it differ from linear regression?",
    "Explain why polynomial regression is considered a special case of multiple linear regression.",
    "How do you determine the degree of the polynomial to use in polynomial regression analysis?",
    "Discuss the process of transforming a linear model into a polynomial regression model. What are the steps involved in this transformation?",
    "What criteria or techniques can be used to select the optimal degree for a polynomial regression model?",
    "Describe the limitations and challenges of polynomial regression. How does the complexity of the model affect its practicality?",
    "Provide examples of real-world problems where polynomial regression is an appropriate modeling technique. Why is it preferred in these scenarios?",
    "Discuss potential ethical implications and biases in polynomial regression, especially when applied to social and economic data.",
    "How is polynomial regression extended to handle multiple independent variables? What are the challenges associated with multivariate polynomial regression?",

    # KNN
    "What is the k-Nearest Neighbors algorithm, and how does it work for classification tasks?",
    "Explain the concept of distance in k-NN. What distance metrics are commonly used, and how do they affect the algorithm's performance?",
    "How does the choice of 'k' affect the performance of the k-NN algorithm? Discuss the impact of too small or too large 'k' values.",
    "Explain the difference between the standard k-NN and the weighted k-NN approaches. When might one prefer weighted k-NN over the standard method?",
    "What are some techniques to improve the scalability and efficiency of k-NN for large datasets?",
    "Why is feature scaling important in k-NN? How can the absence of feature scaling affect the algorithm's performance?",
    "How does k-NN handle missing values in the dataset? What strategies can be employed to deal with missing data for k-NN?",
    "Provide examples of real-world applications where k-NN has been effectively used. Why is k-NN a good choice for these applications?",
    "Discuss potential ethical implications and biases that can arise when using k-NN in decision-making systems, such as in law enforcement or lending.",

    # linear regression
    "Explain the architecture of the Transformer model. How does it differ from previous sequence-to-sequence models?",
    "Discuss the role of self-attention mechanisms in the Transformer. How do they contribute to its ability to model dependencies in data?",
    "Describe the process of multi-head attention in the Transformer architecture. What advantages does this technique offer over traditional single-head attention?",
    "Explain the significance of positional encoding in the Transformer model. How does it incorporate the order of the sequence into the model?",
    "Discuss the components of the Transformer's encoder and decoder layers. How do they interact to process and generate sequences?",
    "Explain the concept of the feedforward neural network within each layer of the Transformer. How does it contribute to the model's ability to learn representations?",
    "Describe the role of layer normalization and residual connections in the Transformer architecture. How do they enhance model training and performance?",
    "Discuss the scalability and efficiency of the Transformer model. How does it perform in comparison to RNNs and CNNs for sequence tasks?",
    "Explain the impact of the Transformer model on the development of NLP applications. How has it influenced the creation of models like BERT, GPT, and T5?",
    "Describe the challenges and limitations of the Transformer architecture. What are the current areas of research aimed at addressing these issues?",

    # ridge regression
    "What is ridge regression, and how does it address the limitations of ordinary least squares (OLS) regression?",
    "Explain the role of the regularization parameter in ridge regression. How does it influence the model?",
    "Derive the cost function used in ridge regression. How does it differ from the cost function of linear regression?",
    "How is the regularization parameter in ridge regression chosen? Discuss the methods used for selecting the regularization parameter",
    "Compare and contrast ridge regression with LASSO regression. Under what circumstances might one be preferred over the other?",
    "Describe the process of fitting a ridge regression model. What considerations must be taken into account during this process?",
    "Provide examples of real-world problems where ridge regression is effectively used. Why is ridge regression suitable for these problems?",
    "How does ridge regression integrate with other machine learning techniques, such as kernel methods?",

    # lasso regression
    "What is the main difference between Lasso and Ridge regression? How do they achieve regularization differently?",
    "Explain the concept of the L1 norm penalty in Lasso regression. How does it influence the model's coefficients?",
    "How does the choice of the regularization parameter (lambda) affect the performance and interpretability of a Lasso model?",
    "What are the limitations of Lasso regression? When might it not be the best approach?",
    "Can you provide an example where Lasso regression might be helpful in a real-world application? Explain the reasoning behind your choice.",
    "How might the interpretability of a Lasso model be used to gain insights into the underlying relationships between variables?",

    #Bayesian Approaches
    "Explain the Bayesian interpretation of probability. How does it differ from the frequentist interpretation?",
    "Describe the role of prior, likelihood, and posterior distributions in Bayesian inference.",
    "How does Bayesian linear regression differ from traditional linear regression?",
    "Define a Bayesian Network. How does it model the joint probability distribution of a set of variables?",
    "Given a simple Bayesian Network, calculate the probability of an event, illustrating the concept of conditional independence.",
    "What are Gaussian Processes (GPs) in machine learning, and how are they used for regression?",
    "Discuss how the choice of kernel affects the predictions in a Gaussian Process regression model.",
    "Explain the concept of Bayesian optimization. How is it used in hyperparameter tuning of machine learning models?",
    "In what types of machine learning problems is a Bayesian approach particularly beneficial, and why?",
    "Explain the concept of Dirichlet Process and its use in non-parametric Bayesian methods.",
    "How do Variational Inference techniques differ from traditional MCMC methods in Bayesian inference?",
    "Discuss the computational challenges associated with Bayesian methods and possible strategies to mitigate these challenges.",

    #Hyperparameter tuning
    'What are hyperparameters in a machine learning model, and how do they differ from model parameters?',
    "Explain the difference between grid search and random search in hyperparameter tuning. What are the pros and cons of each method?",
    "Explain what simulated annealing is and how it can be applied to hyperparameter tuning.",
    "Discuss how the choice of evaluation metric affects the hyperparameter tuning process. Provide examples of metrics used for classification and regression tasks.",
    "Discuss the concept of multi-objective hyperparameter optimization. What are some scenarios where optimizing for multiple objectives might be necessary?",

    #Performance metrics
    "Explain the difference between accuracy, precision, recall, and F1 score. In what scenarios might one metric be preferred over the others?",
    "How does the ROC curve and the AUC score provide insight into a model's performance, and what are the advantages of using AUC as a performance metric?",
    "Describe the Matthews Correlation Coefficient (MCC) and explain why it might be a more informative metric than accuracy in certain situations.",
    "Explain what the Precision-Recall (PR) curve is and when it is more informative than the ROC curve.",
    "Compare and contrast MSE (Mean Squared Error), RMSE (Root Mean Squared Error), and MAE (Mean Absolute Error). When might one prefer MAE over MSE?",
    "Explain the concept of R-squared and adjusted R-squared. How do they differ and what do they indicate about a model's performance?",
    "How do metrics like precision, recall, and F1 score extend to multiclass classification scenarios?",
    "Explain the use of metrics such as Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (NDCG), and Precision at k (P@k) in the context of ranking and recommendation systems."

]

print(len(questions))
