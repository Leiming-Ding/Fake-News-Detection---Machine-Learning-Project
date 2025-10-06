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

