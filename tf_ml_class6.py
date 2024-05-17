
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# Create a new Jupyter Notebook
nb = new_notebook()

data_mark = """
# ml_class6 NLP Part I

Overall Lead: Xiaowen Kang
Contact: xiaowen@svcamp.com

Kevin: Responsible for code validation, homework assignments, and electronic platform-related tasks.
Contact:kevin.lin@svcamp.com

Marco: Responsible for PowerPoint production and validation.
Contact: marco.zhang@svcamp.com


"""
nb.cells.append(new_markdown_cell(data_mark ))

s1 = """

### ML_Class5 NLP Part I : Introduction to NLP and Basic Concepts Algorithms and Foundations

#### Duration: 2 hours (1.5 hours lecture, 0.5 hours project exercise)

---

### Section 1: Introduction to NLP

**Lecture Content (1.5 hours):**

1. **Definition and Scope of Natural Language Processing**
   - Explore the interdisciplinary aspects combining linguistics, computer science, and artificial intelligence.
   - Applications of NLP: voice recognition, text analytics, and automated translation.

2. **Historical Context and Evolution of NLP**
   - **Alan Turing (1912–1954)**: Proposed the Turing Test as a measure of machine intelligence, foundational to the development of AI and NLP.
   - **Noam Chomsky (1928–)**: Introduced the concept of generative grammar in the 1950s, revolutionizing syntactic theories. Main contributions include the Chomsky hierarchy and transformational grammar.
   - **Shift from Rule-based to Statistical Methods**: The introduction of Hidden Markov Models and later, neural networks, marked significant transitions in NLP methodologies during the 1980s and post-2000s.
   - **Latent Dirichlet Allocation (LDA) Background**:
     - Developed by David Blei, Andrew Ng, and Michael I. Jordan in 2003. LDA addresses the challenges of discovering thematic patterns in large text corpora, allowing for the unsupervised classification of documents into topics.
     - **David M. Blei (1976–)**: Significantly contributed to the field of machine learning in topics related to topic modeling and Bayesian data analysis.

   - **Vector Space Model Originator**:
     - The concept of vector space in information retrieval was popularized by Gerard Salton during the 1970s.
     - **Gerard Salton (1927–1995)**: Known as the father of modern search technology, his work laid the groundwork for many of the algorithms used in search engines today.

3. **Applications and Case Studies**
   - **Healthcare**: Use of NLP in extracting patient information from unstructured data to improve diagnosis and treatment plans.
   - **Finance**: Sentiment analysis tools to gauge market sentiment and predict stock movements.
   - **Customer Service**: Chatbots in banking that handle thousands of customer interactions daily, significantly reducing operational costs and improving customer experience.

---

### Section 2: Algorithms and Foundations


**Algorithms Summary Table:**

| Algorithm             | Developer/Contributor              | Year               | Application              | Key Features                                                   |
|-----------------------|------------------------------------|--------------------|--------------------------|----------------------------------------------------------------|
| LDA                   | David M. Blei, Andrew Ng, Michael I. Jordan | 2003               | Topic Modeling           | Discovers hidden thematic structures in large text corpora.    |
| Vector Space Model    | Gerard Salton                      | 1970s              | Information Retrieval    | Uses geometric vectors to represent text documents for similarity assessment. |
| Fourier Transform     | Jean-Baptiste Joseph Fourier       | 1822               | Signal Processing        | Transforms signals between time and frequency domains.         |
| Naive Bayes           | Thomas Bayes                       | 1763               | Probabilistic Classification | Applies Bayes' Theorem with strong independence assumptions.  |
| N-grams               | Markov (Andrey Markov)             | Early 20th Century | Predictive Text Modeling | Predicts the next item in a sequence as a function of the previous ones. |
| Logistic Regression | David Cox                   | 1958         | Classification       | Models the probability of a binary outcome based on input variables.** |




#### Table of NLP Algorithms and Mathematical Theories

1. **Linguistic Foundations**
   - **Syntax, Semantics, and Pragmatics**: Definitions, significance, and their roles in natural language understanding.
   - **Morphological Analysis and POS Tagging**:
     - Formula for POS Tagging using probabilistic models:
       $$
       P(\\text{tag} | \\text{word}) = \\frac{P(\\text{word} | \\text{tag}) \\cdot P(\\text{tag})}{P(\\text{word})}
       $$

2. **Mathematical Foundations**
   - **Vector Space Models and Dot Products**:
     - Discussion on how text is converted into vectors and the role of cosine similarity in determining text similarity.
     - Dot product for cosine similarity:
       $$
       \\text{cosine similarity} = \\frac{\\vec{a} \\cdot \\vec{b}}{\\|\\vec{a}\\| \\|\\vec{b}\\|}
       $$

   - **Fourier Transforms in NLP**:
     - Use in processing signals (e.g., speech analysis) and transforming text data for pattern recognition.
     - Basic Fourier Transform formula:
       
    $$
    X(k) = \\sum_{n=0}^{N-1} x(n) \\cdot e^{-i2\\pi kn/N}
    $$

3. **Core NLP Algorithms and Techniques**
   - **N-grams and Language Modeling**:
     - Mathematical formulation for calculating probabilities of word sequences, critical for models like N-grams:
   $$
   P(w_n|w_{n-1}) = \\frac{C(w_{n-1}, w_n)}{C(w_{n-1})}
   $$
   - **Naive Bayes for Text Classification**:
     - Explanation of applying Bayes' Theorem to classify text, with emphasis on assumptions and limitations.

4. **Latent Dirichlet Allocation (LDA)**
   - **Definition and Importance**: LDA is a topic model used to automatically identify themes in large collections of documents. It assumes documents are composed of latent topic distributions, each defined by a probability distribution of words.
   - **Mathematical Model**:
     
     $$
     P(\\text{topic} | \\text{document}) = \\frac{P(\text{document} | \\text{topic}) \cdot P(\\text{topic})}{P(\\text{document})}
     $$

   - **Application Example**: Used in document classification and information retrieval to improve the relevance of search results by identifying the themes present in a collection of documents.


5. **Logistic Regression Explanation**:

Logistic Regression models the probability \( P(y|x) \) of a dependent binary variable \( y \), given independent variables \( x \). It uses the logistic function:

$$
P(y = 1|x) = \\frac{1}{1 + e^{-(\\beta_0 + \\beta_1x_1 + \\cdots + \\beta_nx_n)}} 
$$


- Where \( \\beta_0, \\beta_1, ..., \\beta_n \) are the coefficients, adjusting these based on the data helps predict outcomes like spam detection or sentiment classification.

### Section 2-2: Algorithm Comparison and Analysis

This section provides a detailed comparison and analysis of Fourier Transforms, Bayesian probability, and vector space models in NLP. It includes mathematical principles, typical application cases, and how these techniques complement each other to enhance the performance and accuracy of NLP tasks.

- **Fourier Transform in NLP**: Mainly used for processing audio signals, identifying and enhancing key frequency components through frequency analysis.
- **Bayesian Probability in NLP**: Widely used in text classification, such as spam detection and sentiment analysis, by calculating the probability of words appearing and their conditional probabilities.
- **Vector Space Model in NLP**: Mainly used for assessing text similarity, such as in document retrieval and clustering, by calculating the cosine similarity between documents.
- **Logistic Regression in NLP**: Used for binary classification tasks such as spam detection, where it models the likelihood of an email being spam based on features like word frequencies.


- **Logistic Regression** vs **Naive Bayes**:
  - Logistic Regression provides a direct probability estimate, which is beneficial in scenarios requiring clear decision boundaries.
  - Naive Bayes, based on applying Bayes' Theorem with strong (naive) independence assumptions, excels in situations with distinct class probabilities especially in text classification.
- Both methods require careful feature selection and preprocessing to perform optimally.


### Part 1: Detailed Text Analysis
**Text**: "The benefits of apples are extensive. Besides being a source of nutrition, there's an old saying, 'An apple a day keeps the doctor away.'"

#### Analysis Using NLP Techniques:

1. **N-grams**:
   - This technique will analyze word sequence dependencies. For example, it could predict the likelihood of "benefits" following "The", and "apples" following "of". This is useful for understanding and generating text based on the sequence learned from this corpus.

2. **Naive Bayes**:
   - If used for sentiment analysis, this technique would assess the overall sentiment of the text as positive based on the presence of positive words like "benefits" and "nutrition". It assumes the independence of each word contributing to the sentiment.

3. **Fourier Transform**:
   - Not typically used directly for textual data, but if this text were converted to speech, Fourier Transform could analyze its audio frequencies to assist in speech recognition tasks.

4. **Bayesian Probability**:
   - Could be used to calculate the probability of specific words or phrases appearing within the text based on their previous occurrences, which helps in text classification or topic modeling.

5. **Vector Space Model**:
   - This model would convert the text into a vector space representation, allowing for the computation of similarities with other documents. Useful in document retrieval systems to find documents with similar content.

6. **Logistic Regression**:
   - Could classify the text's sentiment as positive by modeling the impact of positively connotated words like "benefits" and "extensive".

### Part 2: Sentence Analysis
**Sentence**: "I want to eat an apple."

#### Analysis Using NLP Techniques:

1. **N-grams**:
   - Would predict the probability of each subsequent word based on the previous one(s). For example, after "I want", it might predict "to" as a likely next word based on training data.

2. **Naive Bayes**:
   - For sentiment analysis, it might analyze the sentiment as neutral or slightly positive, focusing on the presence of neutral words like "want" and "eat" and the positive connotation of "apple".

3. **Fourier Transform**:
   - More relevant if the sentence is spoken. It would analyze the frequency components of the spoken sentence to aid in voice recognition.

4. **Bayesian Probability**:
   - Useful in predicting the likelihood of the word "apple" appearing in dietary-related contexts or in predictive typing technologies.

5. **Vector Space Model**:
   - Translates the sentence into a vector form and could be used to find documents that discuss similar topics like eating fruit or healthy habits.

6. **Logistic Regression**:
   - Might analyze the sentiment as neutral, focusing on the commonality of the words with neutral sentiment in the training data.

### Comparative Analysis:
- **N-grams and Naive Bayes** are both probabilistic but differ in application: N-grams are better for understanding and generating text based on sequences, while Naive Bayes is typically applied to classification tasks.
- **Fourier Transforms** are less directly applicable to text unless it is transformed into an audio format.
- **Bayesian Probability and Vector Space Models** provide foundational techniques for a variety of NLP applications from topic modeling to document similarity assessment.

This analysis not only showcases how each method would handle the given text and sentence but also highlights their strengths and limitations in practical scenarios, illustrating the broad range of methodologies in NLP and their specific applications depending on the context and the nature of the data being analyzed.

"""
nb.cells.append(new_markdown_cell(s1))


s3 = """

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
- The overall probability of any email being spam is 30% (i.e., \( P(\\text{spam}) = 0.3 \)) and non-spam is 70% (i.e., \( P(\text{non-spam}) = 0.7 \)).

**Numerator for Spam**:
- \( P(\\text{"FREE"} | \\text{spam}) = 0.8 \)
- \( P(\\text{spam}) = 0.3 \)
- Product: \\( 0.8 \\times 0.3 = 0.24 \)

**Numerator for Non-Spam**:
- \( P(\\text{"FREE"} | \\text{non-spam}) = 0.1 \)
- \( P(\\text{non-spam}) = 0.7 \)
- Product: \( 0.1 \\times 0.7 = 0.07 \)

**Denominator** (Total Probability of "FREE"):
- \( P(\\text{"FREE"}) = (0.8 \\times 0.3) + (0.1 \\times 0.7) = 0.24 + 0.07 = 0.31 \)

**Final Probability of Spam given "FREE"**:
- \( P(\\text{spam} | \\text{"FREE"}) = \\frac{0.24}{0.31} \approx 0.774 \)

#### 2. Calculating Probabilities for the Non-Spam Message:
Assuming the message doesn’t contain typical spam words like "FREE," we adjust the probabilities accordingly.

**Numerator for Spam**:
- \( P(\\text{not "FREE"} | \\text{spam}) = 1 - 0.8 = 0.2 \)
- \( P(\\text{spam}) = 0.3 \)
- Product: \( 0.2 \\times 0.3 = 0.06 \)

**Numerator for Non-Spam**:
- \( P(\\text{not "FREE"} | \\text{non-spam}) = 1 - 0.1 = 0.9 \)
- \( P(\\text{non-spam}) = 0.7 \)
- Product: \( 0.9 \\times 0.7 = 0.63 \)

**Denominator** (Total Probability of not "FREE"):
- \( P(\\text{not "FREE"}) = (0.2 \\times 0.3) + (0.9 \\times 0.7) = 0.06 + 0.63 = 0.69 \)

**Final Probability of Spam given not "FREE"**:
- \( P(\\text{spam} | \\text{not "FREE"}) = \\frac{0.06}{0.69} \approx 0.087 \)

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
print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\\n", classification_report(y_test, y_pred))
```

#### Comparison with Another Method: Logistic Regression
We will also implement a Logistic Regression model to compare its performance with the Naive Bayes classifier.



### Logistic Regression Formula Analysis:
$$
P(y=1 | x) = \\frac{1}{1 + e^{-(\\beta_0 + \\beta_1x_1 + \\cdots + \\beta_nx_n)}} 
$$

#### Domain and Range:
- **Domain**: The domain of \( x_1, \dots, x_n \) in this formula is all real numbers (\( \mathbb{R}^n \)), because features extracted from the text (like word frequencies or TF-IDF scores) can take any real value.
- **Range**: The range of \( P(y=1 | x) \) is between 0 and 1. This represents the probability of the email being spam, and probabilities are always in this interval.

#### Behavior of the Exponential Component:
- When the linear combination \( \\beta_0 + \\beta_1x_1 + \\cdots + \\beta_nx_n \) is very large (positive), \( e^{-\\text{(large value)}} \) approaches 0, making the denominator close to 1, and hence \( P(y=1|x) \) approaches 1.
- Conversely, if the linear combination is very large (negative), \( e^{-\\text{(large negative value)}} \) becomes very large, making \( P(y=1|x) \) approach 0.

### Naive Bayes Formula Analysis:
\[ P(y | x_1, \\dots, x_n) = \\frac{P(x_1, \dots, x_n | y) P(y)}{P(x_1, \dots, x_n)} \]

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
print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\\n", classification_report(y_test, y_pred_lr))
```


### Analysis
After executing both models, participants will analyze the results to understand how each model performs in terms of accuracy, precision, recall, and F1-score. The discussion will focus on why certain models perform better and how different data characteristics can influence the outcome of NLP tasks.



"""
nb.cells.append(new_markdown_cell(s3))



data_code_test = """

 # tip_math4_section_hand_on_excise

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('./ml_class6_spam.csv', encoding='latin-1')
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
print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\\n", classification_report(y_test, y_pred))



from sklearn.linear_model import LogisticRegression

# Train Logistic Regression classifier
model_lr = LogisticRegression()
model_lr.fit(X_train_vec, y_train)

# Predict on test data
y_pred_lr = model_lr.predict(X_test_vec)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\\n", classification_report(y_test, y_pred_lr))




"""
nb.cells.append(new_code_cell(data_code_test))


data_code_test2 = """

"""
nb.cells.append(new_code_cell(data_code_test2))






data_code_test3 = """

"""

nb.cells.append(new_code_cell(data_code_test3))


with open('ml_calss6_v1.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

