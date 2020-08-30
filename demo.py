from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
# instead of loading a roberta squad pretrained model
# (for which u just need to replace the long path string by `roberta_large`)
# u just need to instead put the (relative) path to the folder containing the pytorch_model.bin file
tokenizer = RobertaTokenizer.from_pretrained('models/tanda_roberta_large_asnq_wikiqa/ckpt/')
# you need to do it for both tokenizer and the main model, one will not work
model = RobertaForSequenceClassification.from_pretrained('models/tanda_roberta_large_asnq_wikiqa/ckpt/')

context = '''
In machine learning, the hinge loss is a loss function used for training classifiers. The hinge loss is used for "maximum-margin" classification, most notably for support vector machines (SVMs).[1]
For an intended output t = ±1 and a classifier score y, the hinge loss of the prediction y is defined as:
{\displaystyle \ell (y)=\max(0,1-t\cdot y)}\ell(y) = \max(0, 1-t \cdot y) 
Note that {\displaystyle y}y should be the "raw" output of the classifier's decision function, not the predicted class label. For instance, in linear SVMs, {\displaystyle y=\mathbf {w} \cdot \mathbf {x} +b}y = \mathbf{w} \cdot \mathbf{x} + b, where {\displaystyle (\mathbf {w} ,b)}(\mathbf{w},b) are the parameters of the hyperplane and {\displaystyle \mathbf {x} }\mathbf {x}  is the input variable(s).
'''
question = "what is hinge loss function"
inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"].tolist()[0]

text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
answer_start_scores, answer_end_scores = model(**inputs)

answer_start = torch.argmax(
    answer_start_scores
)  # Get the most likely beginning of answer with the argmax of the score
answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

print(f"Question: {question}")
print(f"Answer: {answer}\n")
