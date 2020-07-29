



##Chatbot Text Similarity

---

 Problem statement:
 Identification of chats whether given sample is a similar to other sample short text .
 This dataset contains text pairs extracted from chat logs from 50+ bots with a similar or not similar label. Note that both texts of the pairs are from human users.

 1. The texts in the pairs are text messages written by humans to get some task done with a task oriented chat bot.
 2. Unlike ideal datasets which are mostly generated from news, emails, etc., and contains well structured English sentences, this dataset contains anomalies in sentence structure, spelling mistakes, etc. which are common on messaging/voice interface.
 3. It can be best used to evaluate domain agnostic word representations for semantic similarity downstream task.

 The file `train.csv` has 4 columns
 - `pid`  - Unique id for pairs.
 - `sentence1` - One of messages sent by some user to one of the bots.
 - `sentence2` - Another one of message sent by some user to the same bot as `sentence1`.
 - `label` - Either 0 or 1. `0` means `sentence1` and `sentence2` are not similar. `1` means `sentence1` and `sentence2` are similar

 Approach
 1.	Pre-process your data, split it to test and evaluation.
 2.	Build your model ,  you can use pre-trained models
 3.	Predict on test data
