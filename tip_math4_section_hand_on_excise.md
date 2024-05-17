### Section 3: Hands-On Project Exercise - Building a Simple Spam Detection System Using Naive Bayes

-(0.5 hours)

This structured approach not only provides a historical and theoretical framework but also connects these concepts with practical examples, enhancing the learning experience.

In this practical exercise, participants will develop a basic spam filter using the Naive Bayes algorithm. This project will reinforce the theoretical concepts discussed in the lecture and provide hands-on experience in applying NLP techniques to solve real-world problems. We'll compare the Naive Bayes classifier with another common NLP method to highlight their respective strengths and weaknesses.

#### Dataset
For this exercise, we will use a publicly available dataset of SMS messages that have been labeled as either 'spam' or 'ham' (non-spam). This dataset is commonly used in text processing and provides a good mix of text types and formulations.

#### Python Environment Setup
Participants are expected to have Python installed, along with the following libraries:
- NumPy
- pandas
- scikit-learn
- nltk


### Example Messages:
1. **Ham**: "Sorry, it's a lot of friend-of-a-friend stuff, I'm just now about to talk to the actual guy who wants to buy"
2. **Spam**: "FREE for 1st week! No1 Nokia tone 4 ur mob every week just txt NOKIA to 8007 Get txting and tell ur mates www.getzed.co.uk POBox 36504 W45WQ norm150p/tone 16+"

### Naive Bayes Analysis:

#### 1. Calculating Probabilities for the Spam Message:
Let's assume:
- The word "FREE" appears in 80% of spam emails and 10% of non-spam emails.
- The overall probability of any email being spam is 30% (i.e., \( P(\text{spam}) = 0.3 \)) and non-spam is 70% (i.e., \( P(\text{non-spam}) = 0.7 \)).

**Numerator for Spam**:
- \( P(\text{"FREE"} | \text{spam}) = 0.8 \)
- \( P(\text{spam}) = 0.3 \)
- Product: \( 0.8 \times 0.3 = 0.24 \)

**Numerator for Non-Spam**:
- \( P(\text{"FREE"} | \text{non-spam}) = 0.1 \)
- \( P(\text{non-spam}) = 0.7 \)
- Product: \( 0.1 \times 0.7 = 0.07 \)

**Denominator** (Total Probability of "FREE"):
- \( P(\text{"FREE"}) = (0.8 \times 0.3) + (0.1 \times 0.7) = 0.24 + 0.07 = 0.31 \)

**Final Probability of Spam given "FREE"**:
- \( P(\text{spam} | \text{"FREE"}) = \frac{0.24}{0.31} \approx 0.774 \)

#### 2. Calculating Probabilities for the Non-Spam Message:
Assuming the message doesn’t contain typical spam words like "FREE," we adjust the probabilities accordingly.

**Numerator for Spam**:
- \( P(\text{not "FREE"} | \text{spam}) = 1 - 0.8 = 0.2 \)
- \( P(\text{spam}) = 0.3 \)
- Product: \( 0.2 \times 0.3 = 0.06 \)

**Numerator for Non-Spam**:
- \( P(\text{not "FREE"} | \text{non-spam}) = 1 - 0.1 = 0.9 \)
- \( P(\text{non-spam}) = 0.7 \)
- Product: \( 0.9 \times 0.7 = 0.63 \)

**Denominator** (Total Probability of not "FREE"):
- \( P(\text{not "FREE"}) = (0.2 \times 0.3) + (0.9 \times 0.7) = 0.06 + 0.63 = 0.69 \)

**Final Probability of Spam given not "FREE"**:
- \( P(\text{spam} | \text{not "FREE"}) = \frac{0.06}{0.69} \approx 0.087 \)

### Conclusion:
- **With "FREE" in the message**, the Naive Bayes classifier shows a high probability of being spam (approximately 77.4%).
- **Without "FREE" in the message**, the probability significantly drops (approximately 8.7%), indicating it's likely not spam.

These examples illustrate how Naive Bayes uses word probabilities conditioned on the class (spam or non-spam) to determine the overall likelihood of an email belonging to a certain class based on the presence or absence of certain keywords.


#### Code: Spam Detection Using Naive Bayes
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Preprocess data
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Convert text to vectors
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict on test data
y_pred = model.predict(X_test_vec)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

#### Comparison with Another Method: Logistic Regression
We will also implement a Logistic Regression model to compare its performance with the Naive Bayes classifier.



### Logistic Regression Formula Analysis:
$$
 P(y=1 | x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + \cdots + \beta_nx_n)}} 
$$

#### Domain and Range:
- **Domain**: The domain of \( x_1, \dots, x_n \) in this formula is all real numbers (\( \mathbb{R}^n \)), because features extracted from the text (like word frequencies or TF-IDF scores) can take any real value.
- **Range**: The range of \( P(y=1 | x) \) is between 0 and 1. This represents the probability of the email being spam, and probabilities are always in this interval.

#### Behavior of the Exponential Component:
- When the linear combination \( \beta_0 + \beta_1x_1 + \cdots + \beta_nx_n \) is very large (positive), \( e^{-\text{(large value)}} \) approaches 0, making the denominator close to 1, and hence \( P(y=1|x) \) approaches 1.
- Conversely, if the linear combination is very large (negative), \( e^{-\text{(large negative value)}} \) becomes very large, making \( P(y=1|x) \) approach 0.

### Naive Bayes Formula Analysis:
\[ P(y | x_1, \dots, x_n) = \frac{P(x_1, \dots, x_n | y) P(y)}{P(x_1, \dots, x_n)} \]

#### Domain and Range:
- **Domain**:
  - \( y \) is binary, taking values in {0, 1}, where 0 might represent 'non-spam' and 1 'spam'.
  - \( x_1, \dots, x_n \) are the features extracted from the email, similar to logistic regression, covering all real numbers depending on how the features are defined and normalized.
- **Range**: Similar to logistic regression, the range of \( P(y | x_1, \dots, x_n) \) is between 0 and 1, as it represents a probability.

#### Parameters and Variables:
- \( y \) represents the class label (spam or not spam).
- \( x_1, \dots, x_n \) represent the features (words, phrases, other characteristics).
- \( P(x_1, \dots, x_n | y) \) is the likelihood of seeing the particular pattern of features given the class.
- \( P(y) \) is the prior probability of the email being spam or not spam, based on the training dataset.
- \( P(x_1, \dots, x_n) \) is the evidence or the overall likelihood of observing that particular pattern of features under any class.

These formulas underscore the statistical nature of spam detection using logistic regression and Naive Bayes, where each feature's contribution to the final decision is calculated differently under each model. Logistic regression computes a weighted sum of features, while Naive Bayes computes probabilities based on the statistical independence of features. Both are effective for spam detection, but their performance might differ based on the dataset's characteristics and the specific features used.



```python
from sklearn.linear_model import LogisticRegression

# Train Logistic Regression classifier
model_lr = LogisticRegression()
model_lr.fit(X_train_vec, y_train)

# Predict on test data
y_pred_lr = model_lr.predict(X_test_vec)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
```


### Analysis
After executing both models, participants will analyze the results to understand how each model performs in terms of accuracy, precision, recall, and F1-score. The discussion will focus on why certain models perform better and how different data characteristics can influence the outcome of NLP tasks.



