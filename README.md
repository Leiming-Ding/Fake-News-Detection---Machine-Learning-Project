This project is to develop a machine learning model to detect fake news.

The dataset is from Kaggle: https://www.kaggle.com/datasets/rajatkumar30/fake-news/data

The original datset includes three variables: title, text, and label

Fifteen engineered features include:
- `title_length`, `text_length`, `title_num_chars` (text length metrics)
- `text_num_chars` (number of characters in the text)
- `reading_time` (estimated reading duration)
- `average_grade_level` (readability level)
- `num_urls` (number of URLs in the text)
- `has_pronoun`
- `has_question_or_exclaim`
- `num_mentions`, `num_hashtags`
- `avg_sentiment` (average sentiment score)
- `title_text_similarity`
- `readability` (0-1)
- `num_function_words`

I tested six models: logistic regression, decision tree, K-nearest neighbors, random forest, XGBoost, and support vector machine. 
For each model, I tested different combinations of parameters and found the best parameters for the highest training score.
Finally, I compared precision, recall, F1 score, and accuracy for each model.


Project Description

Purpose of this project

Nowadays, people acquire information from different sources every day, including social media, websites, magazines, etc. However, it does not mean that people become more informed with that abundant information. Misinformation, including fake news, negatively impacts people's lives via distorting people's understanding or stirring up unpleasant emotions or feelings. In this project, I intended to develop a machine learning algorithm to classify fake news. This is a classification task and a supervised learning algorithm. I fitted multiple models, including logistic regression, decision tree, K-nearest neighbors, random forest, XGBoost, and support vector machine, and then compared the results. 

Goal of this project

Misinformation is now everywhere, possibly causing great harms to the society and people. For example, false information may polarize the society, eroding trust in science and government agencies. It may also decrease people's health by spreading wrong information. The goal is to develop an algorithm to indentify the fake news or fake information. Social media platform may integrate this algorithm to flag or de-prioritize the unreliable information. This algorithm can also contribute to maintaining a healthy information ecosystem. I used the dataset from Kaggle including three variables: title of the news, content of the news and the label of the news. I use natural language processing skills to engineer 15 features. I intended to know whether these features could predict the status of fake news and how about the performance of different models.

Dataset Description

The reference of this dataset is an open dataset from Kaggle. Kumar, R. (n.d.). Fake news prediction dataset [Data set]. Kaggle. https://www.kaggle.com/datasets/rajatkumar30/fake-news/data  There are three variables in the dataset: title, text, and label. I engineered 15 variables: title_length, text_length, title_num_chars (the number of characters in the title), text_num_chars, reading_time (the expected time spent on reading the news), average_grade_level (the grade level that the content is suitable for), num_urls (the number of url links in the content), has_pronoun, has_question_or_exclaim, num_mentions (number of @s), num_hashtags, avg_sentiment (average of sentiments), title_text_similarity, readability (the readability score of the text), num_function_words.

Data

This is tabulated data. There are 6276 observations (20 observations deleted due to missing data). There are 38 variables in the dataset, including 1 title variable, 1 content variable, 1 label variable, and 35 engineered variables. Among the 35 enginnered variables, There are 2 categorical variables and 33 continuous variables. Some key features include title_text_similarity, which means the similarity (ranging from 0 to 1) of title and text, and text_length, which means the length of the content. This is not a multi-table form or not gathered from many data sources.

Data Cleaning

Before analysis, I first conducted a learning of the data. I first looked at which observations had missing values and then dropped those missing values. Finally, I checked the dataset again to make sure each coloum is free of variables. I finally removed 20 observations and then checked the balance of the dataset (fake new and true news are around 50%). Because I engineered all the variables, it is not hard to do the data clearning. I expect that some columns in other datasets may contain a lot of missing values and some imputation methods may be needed. 

Exploratory Data Analysis

I generated the correlation heatmap and the distribution of variables. Based on the correlation heatmap, title_text_similarity had the highest correlation with the label, 0.31. Text_length, text_num_chars, reading_time, and num_function_words have a negative correlation of 0.15 with the label. There are some highly correlated variables. For example, num_function_words had a high correlation with text_length, text_num_chars, and reading_time, which is expected because when the text is long, the number of function words in the text will also increase. Because I engineered all the variables, it was possible some of them represented similar meanings. One challenge will be to decide which features are important at this step. However, I decided to keep all of them to fit the following models

Models

I developed six models: logistic regression, decision tree, K-nearest neighbors, random forest, XGBoost, and support vector machine. I tried the logistic regression model as a baseline model. However, it fails to capture the nonlinear relationships. Therefore, I tried other models. For each model, I set up algorithms to tune the parameters. For example, for logistic regression, I tried Lasso and Ridge regularization to let the model choose the best result. For each method, I also use cross validation methods to split the training into five folds. 

Results and Analysis

For results, I paid more attention to the fake news detection. So I focused on the precision and recall of the fake news category. Precision means the percentage of fake news among all the predicted fake news. Recall means the percentage of fake news the model detected among all the fake news. I also included F1 score and the accuracy of the whole model

For precision / Recall / F1 score / Accuracy
##logistic regression        0.75     /        0.67      /        0.71         /     0.72
##decision tree              0.75     /        0.73      /        0.74         /     0.74
##KNN                        0.82     /        0.64      /        0.72         /     0.75
##Random Forest              0.79     /        0.76      /        0.77         /     0.78
##XGBoost                    0.82     /        0.73      /        0.77         /     0.79
##Support vector machine     0.79     /        0.74      /        0.77         /     0.77

From the above model results, random forest, XGBoost, and support vector machine had the comparatively better combination of results. XGboost had the highest precision and random forest had the highest recall.For the fake news class (coded as 1), precision ranged from 0.75 to 0.82 across models, while recall ranged from 0.64 to 0.76.These results highlight that ensemble and kernel-based methods more effectively identify fake news while maintaining high classification precision.

Discussion

Takeaways: These machine learning models worked fairly well just depending on 15 engineered variables. I tried different combinations of parameters in my model building, which took longer time than expected. Some models, like random forest, has a good performance and also can show the importance of features, which may help with the interpretation. Why some models worked well: Random forest and XGBoost are both ensemble methods, which may lead to its better performance. SVM based on kernel is another advantage over other models. Decision tree can be easily overfitting, which may cause a lower performance in the testing dataset. KNN is distance-based and it may not work well in some high-dimensional feature spaces.

Ways to improve: I would suggest some deep learning models, which can better capture the complex information of the data. They may work better in these text_based tasks. More variables can be engineered based on theory to improve the model performance.
