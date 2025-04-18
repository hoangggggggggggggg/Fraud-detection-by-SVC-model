# Credit-Card-Fraud-Detection-Using-Support-Vector-Machine

 ## ABSTRACT
Credit card fraud is a major issue, causing billions of dollars in losses annually. Machine learning can help detect credit card fraud by recognizing patterns that indicate fraudulent transactions. Credit card fraud involves the physical loss of a credit card or the compromise of sensitive card information. Various machine learning algorithms can be employed for fraud detection. In this project, a Support Vector Machine (SVM) model will be developed to identify credit card fraud. The model will be trained using a dataset containing historical credit card transactions and tested on a separate dataset of unseen transactions. The performance of the SVM model will be evaluated and compared to determine its effectiveness in detecting fraudulent transactions.
<br>

<br>

## Overview
Payment fraud continues to pose a significant threat to financial institutions and consumers worldwide. In 2023, global losses from credit and debit card fraud were estimated to reach $32.34 billion, demonstrating a steady increase from $30.06 billion in 2022 (Statista, 2023). In the United States, a report by the Association for Financial Professionals (AFP) indicated that 65% of organizations fell victim to payment fraud in 2022, with 71% of those affected by business email compromise (BEC) schemes, highlighting the increasing sophistication of fraud tactics (AFP, 2023). Similarly, the United Kingdom experienced severe financial losses due to fraud, with over £1.2 billion stolen in 2022, including £726.9 million lost through unauthorized fraud and £485.2 million through authorized push payment (APP) fraud (UK Finance, 2023). These alarming figures emphasize the growing need for intelligent systems capable of detecting and preventing fraudulent transactions.

In response to this challenge, this project proposes the development of a machine learning model using Support Vector Machine (SVM) to detect credit card fraud by identifying anomalies and suspicious patterns in transactional data. By leveraging historical transaction records and applying the SVM classification algorithm, the model aims to improve fraud detection accuracy and contribute to the financial security of consumers and institutions.
<br>
<br>

## Project goals
The main aim of this project is to detect fraudulent credit card transactions, as it is crucial to identify fraudulent transactions to prevent customers from being charged for products they did not purchase. Fraudulent transactions will be detected using the machine learning technique Support Vector Machine (SVM). After applying SVM, the model's performance will be evaluated to determine its accuracy and effectiveness in detecting fraudulent transactions. Graphs and numerical data will be provided to illustrate the performance of the SVM model. Additionally, the project explores previous research and techniques used to identify fraud within a dataset.
<br>
<br>

## Data Source

The dataset was retrieved from an open-source website, Kaggle.com. It contains data on transactions made in 2013 by European credit card users in two days only. Thedataset consists of 31 attributes and 284,808 rows. Twenty-eight attributes are numeric variables that, due to the confidentiality and privacy of the customers, have been transformed using PCA transformation; the three remaining attributes are ”Time”, which contains the elapsed seconds between the first and other transactions
of each Attribute, ”Amount” is the amount of each transaction, and the final attribute “Class” which contains binary variableswhere “1” is a case of fraudulent transaction, and “0” is not as case of fraudulent transaction.
<br>
<br>
<b>Dataset: </b>
<a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud">kaggle Dataset</a>

<br>

<br>

## Conclusion
This project successfully implemented a Support Vector Machine (SVM) model to detect fraudulent credit card transactions. Using a real-world dataset from Kaggle, the model was trained and tested to identify suspicious patterns and anomalies in transaction behavior.

Despite the significant class imbalance in the dataset, the SVM model performed well, achieving the following evaluation metrics:

Accuracy: 0.97 

Precision: 1.00 

Recall: 0.90

F1-score: 0.95

AUC : 0.9927190688612009

These results demonstrate that SVM is an effective approach for fraud detection, capable of distinguishing between legitimate and fraudulent transactions. Overall, the project highlights the potential of machine learning in enhancing payment security and provides a foundation for future work, including more advanced models, real-time detection systems, and better handling of imbalanced data.

