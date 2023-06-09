Text Summarization using Transformers
Summarization is a method for shortening a text without losing its essential content. It's a hot topic in Natural Language Processing (NLP). There are two types of summarization methods, depending on whether or not the original text's sentence structure is preserved. New sentences are generated using NLP techniques, extractive, and abstractive summarization.

Extractive Text Summarization: In extractive summarization, the most significant sentences from an article are chosen and put logically. The sentences used to create the summary are taken verbatim from the original text.
Abstractive Text Summarization: An NLP task aims to generate a concise summary of a source text. Abstractive summarization does not simply copy essential phrases from the source text but also potentially develops new relevant phrases, which can be seen as paraphrasing.
BART for Summarization of Text Data
BART stands for Bidirectional and Auto-Regressive Transformer. Its primary features are a bidirectional encoder from BERT and an autoregressive decoder from GPT. The encoder and decoder are united using the seq2seq model to form the BART algorithm. Let us look at it in more detail.

BART Model Architecture.
To understand the BART transformer, one needs to closely look at BERT and GPT. BERT performs the Masked Language Modelling with the help of its bidirectional transformer and predicts the missing values. On the other hand, GPT uses its autoregressive decoder to predict the next token in a sentence. Merging both of these results in the BART model, as depicted in the image below.

BART Pre-training
There are five primary methods for training BART with noisy text:

Token Masking: Randomly, a small number of input points are masked.
Token Deletion: Certain tokens from the document are deleted.
Text Infilling: Multiple tokens are replaced with a single mask token.
Sentence Permutation: Sentences are identified with the help of ‘.’ and are then shuffled for training.
Document Rotation: A token is randomly selected, and the sequence is rotated so that the document begins with the chosen token. These strategies augment the dataset and make the BART model better understand the natural language.
BART Fine-Tuning Down Stream Tasks
Depending on the task one wants to perform using BART, they can fine-tune the model as discussed in the section below:

Sequence classification: To perform sequence classification using BART, we feed the same input to the encoder and the decoder. The final decoder token's final hidden state is fed into a new multi-class linear classifier.
Token classification: For solving classification problems using BART, the complete document is passed into the encoder and decoder, and the top hidden state of the decoder is used as a representation for each word. One then uses this representation for the classification of tokens.
Sequence generation: As an autoregressive decoder is a part of the BART model’s architecture, we can use it for sequence generation problems. The input at the encoder acts as the input, and the decoder generates the output autoregressively.
Machine translation: Unlike other state-of-the-art models, BART combines both an encoder and a decoder, making it suitable for English translation. To add a new set of encoder parameters (learn using bitext) to the model and use BART as a single pre-trained decoder for machine translation.Dataset for Text
Dataset for Text Summarization using BART
The data used is from the curation base repository, which has a collection of 40,000 professionally written summaries of news articles, with links to the articles themselves. The data was downloaded in the form of a CSV file and has the following features:

Article titles – title for the texts
Summaries – Summary for each text
URLs – the URL links
Dates
Article content – content under each article
Aim of the Bart Text Summarization Project
The BART model will perform abstractive text summarization in Python on a given text data

Tech Stack
Language - Python
Libraries - pandas, sklearn, PyTorch, transformers, PyTorch Lightning
Environment – Google Colab
BART Summarization Project: Solution Approach
Import the dataset from the dataset library and load a subset of the data. (To get an overview of the summarized data)
Create a class function for the dataset.
Create a class function for the BART data loader.
Create an abstractive summarization model class function.
Create a BART tokenizer
Define the data loader
Read and prepare the data.
Perform train test split.
Create the main class that runs the ‘BARTForConditionalGeneration’ model and tokenizer as an input.
Define the trainer class and then fit the model.
For running the web application:
Upload the run_app.ipynb file on your drive, open it with colab and follow the steps mentioned:

Install dependencies
Setup ngrok
Signup with ngrok and copy auth code and paste to create config file
zip output folder and upload it the notebook
Unzip it (command mentioned it will create a directory)
Move to the output directory
And then run app.py with the command !python app.py.
