from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

tokenizer = RobertaTokenizer.from_pretrained('models/tanda_roberta_large_asnq_wikiqa/ckpt/')
model = RobertaForSequenceClassification.from_pretrained('models/tanda_roberta_large_asnq_wikiqa/ckpt/')

inputs = tokenizer("what is machine learning? Machine learning is a subbranch of artificial intelligence", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
loss, logits = outputs[:2]
print(loss, logits)

inputs = tokenizer("what is machine learning? deep learning is fun but complicated", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
loss, logits = outputs[:2]
print(loss, logits)
