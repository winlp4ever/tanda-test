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
df = pd.read_csv('web-answers.csv')
nb_rows = df.shape[0]

tanda_loss = [0.0 for _ in range(nb_rows)]
tanda_answer_sentence = ['' for _ in range(nb_rows)]
tanda_sentence_loss = [0.0 for _ in range(nb_rows)]
tanda_sentence_proba = [0.0 for _ in range(nb_rows)]


def getLossAndProbas(q, a, model, labels):
    try:
        inputs = tokenizer.encode_plus(q, a, add_special_tokens=True, return_tensors="pt")
        outputs = model(**inputs, labels=labels)
        loss, logits = outputs[:2]
        probas = torch.sigmoid(logits).detach().numpy()
        return loss.detach().numpy(), probas[0, 1]
    except:
        print('Too long answer!!!')
        return 10, 0


def divideToSentences(doc, nlp):
    try:
        doc = nlp(doc)
        sentences = [sent.string.strip() for sent in doc.sents]
        return sentences
    except:
        return []

for i, r in df.iterrows():
    print('{}: Working on Question: {} - qid: {} - aid: {}'.format(i, r['question'], r['qid'], r['aid']))
    loss = 10 # default loss value
    try:
        loss, _ = getLossAndProbas(r['question'], r['answer'], model, labels)
    except:
        print('too long paragraph')
    tanda_loss[i] = loss
    sens = divideToSentences(r['answer'], nlp)
    mnsen, mn, probas = '', 10, 0
    for sen in sens:
        ls, ps = getLossAndProbas(r['question'], sen, model, labels)
        if ls < mn:
            mn = ls
            probas = ps
            mnsen = sen
    tanda_answer_sentence[i] = mnsen
    tanda_sentence_loss[i] = mn 
    tanda_sentence_proba[i] = probas

df['tanda_loss'] = tanda_loss
df['tanda_answer_sentence'] = tanda_answer_sentence
df['tanda_sentence_loss'] = tanda_sentence_loss
df['tanda_sentence_proba'] = tanda_sentence_proba

df.to_csv('tanda-web-results.csv')