import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import spacy

# load tanda roberta model
tokenizer = RobertaTokenizer.from_pretrained('models/tanda_roberta_large_asnq_wikiqa/ckpt/')
model = RobertaForSequenceClassification.from_pretrained('models/tanda_roberta_large_asnq_wikiqa/ckpt/') 
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1

# load spacy english language model
nlp = spacy.load('en_core_web_lg')

# load csv
df = pd.read_csv('ds-qa.csv')

nb_rows = df.shape[0]

tanda_loss = [0.0 for _ in range(nb_rows)]
tanda_answer_sentence = ['' for _ in range(nb_rows)]
tanda_sentence_loss = [0.0 for _ in range(nb_rows)]
tanda_sentence_proba = [0.0 for _ in range(nb_rows)]

def getLossAndProbas(q, a, model, labels):
    inputs = tokenizer(q + '? ' + a, return_tensors="pt")
    outputs = model(**inputs, labels=labels)
    loss, logits = outputs[:2]
    return loss.numpy(), torch.sigmoid(logits).numpy()

def divideToSentences(doc, nlp):
    doc = nlp(raw_text)
    sentences = [sent.string.strip() for sent in doc.sents]
    return sentences

for i, r in df.iterrows():
    print('Working on Question: {} - qid: {} - aid: {}'.format(r['question'], r['qid'], r['aid']))
    loss, _ = getLossAndProbas(r['question'], r['answer'], model, labels)
    tanda_loss[i] = loss
    sens = divideToSentences(r['answer'], nlp)
    mnsen, mn, probas = '', 10, [1, 0]
    for sen in sens:
        ls, ps = getLossAndProbas(r['question'], sen, model, labels)
        if ls < mn:
            mn = ls
            probas = ps
            mnsen = sen
    tanda_answer_sentence[i] = mnsen
    tanda_sentence_loss[i] = mn 
    tanda_sentence_proba[i] = 1-probas[0]

df['tanda_loss'] = tanda_loss
df['tanda_answer_sentence'] = tanda_answer_sentence
df['tanda_sentence_loss'] = tanda_sentence_loss
df['tanda_sentence_proba'] = tanda_sentence_proba

df.to_csv('tanda-results.csv')

