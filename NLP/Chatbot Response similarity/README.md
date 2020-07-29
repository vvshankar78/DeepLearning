



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

 Problem statement is that we would like to classify if the 2 texts are similar or not. Since the training data is already labels, this makes it a supervised learning classification problem.

 Approach: The approach I am using is that I am concating both the sentences and making it as a single sentence. This is captured when we build a datablock using fastai. It concats the sentenses with a marker.

 Also, Fast ai takes care of the preprocessing like tokenization, numericalization creating databunch.

 Model:
 Language Model: learn the language using a pre-trained model(wiki) and add the chat sentences to it and create an updated language model.

 Classification model: build a classification model based on the vocab used in language model
