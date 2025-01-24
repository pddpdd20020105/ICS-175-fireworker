---
layout: default
title: Proposal
---

# {{ page.title }}: Text Sentiment Analysis System


### Summary of Project
People’s comments are usually hard to understand so it is very complex for company or other users to know what exactly the comments are saying. We are doing emotion/mood detectors from text. This can be used to decide whether a comment is positive or negative. It will be a great tool for market research or auto-deleting malicious comments from comment sections. The input for our system is a plain text, the output will be one of the following `POSITIVE`, `NEGATIVE`, `NEUTRAL`.
<br>
<br>

### AI/ML Algorithms<br />
1. Using Logistic Regression as a baseline model.<br />
2. We plan to use LSTM or BERT for further experiments as a deep learning model to improve the performance.
<br>
<br>

### Evaluation Plan<br />
Quantitative Evaluation<br />

To evaluate the success of our sentiment analysis system, we’ll focus on metrics like accuracy, precision, recall, and F1-score to see how well it classifies text into positive, negative, or neutral sentiments. Our baseline model will use Logistic Regression with TF-IDF or Bag-of-Words for feature extraction, and we expect it to achieve an F1-score of around 70-75% on standard datasets. For the advanced models, like LSTM or BERT, we’re aiming for an F1-score improvement to about 80-85%. We’ll test the models on popular datasets like the IMDB Sentiment Dataset or Twitter Sentiment Analysis Dataset, splitting the data 80-20 for training and testing and using cross-validation to make sure our results are consistent and reliable.

Qualitative Evaluation<br />

To make sure the system actually works, we’ll run it on some simple examples, like “I love this product” (positive) or “This was the worst experience ever” (negative), and see if it predicts the right sentiment. We’ll also test more tricky cases, like “The movie was good, but the ending was disappointing,” to see how it handles mixed sentiments. To visualize what the model is doing, we’ll use tools like attention maps (for BERT) or analyze the hidden states (for LSTM). Our big goal is to create a system that can handle really complicated texts, like medical or legal ones, without needing a lot of extra training. If we can get to that point, we’ll show it off with a live demo where users can input text and see both the predictions and explanations.<br />
Moonshot case is the system has over 95% accuracy which is pretty hard even for a human reader.
<br>
<br>

### Meet the Instructor<br />
1/23/2025 Group meeting at discord.<br />
Decides to meet the instructor at 1/28/2025 at 12:10pm.
<br>
<br>

### AI Tool Usage<br />
Using Chatgpt for brainstorming the ideas.
<br>
<br>

