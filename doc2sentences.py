import spacy

raw_text = '''
In machine learning, the hinge loss is a loss function used for training classifiers. The hinge loss is used for "maximum-margin" classification, most notably for support vector machines (SVMs).[1]

For an intended output t = ±1 and a classifier score y, the hinge loss of the prediction y is defined as

{\displaystyle \ell (y)=\max(0,1-t\cdot y)}\ell(y) = \max(0, 1-t \cdot y)
Note that {\displaystyle y}y should be the "raw" output of the classifier's decision function, not the predicted class label. For instance, in linear SVMs, {\displaystyle y=\mathbf {w} \cdot \mathbf {x} +b}y = \mathbf{w} \cdot \mathbf{x} + b, where {\displaystyle (\mathbf {w} ,b)}(\mathbf{w},b) are the parameters of the hyperplane and {\displaystyle \mathbf {x} }\mathbf {x}  is the input variable(s).

When t and y have the same sign (meaning y predicts the right class) and {\displaystyle |y|\geq 1}|y| \ge 1, the hinge loss {\displaystyle \ell (y)=0}\ell(y) = 0. When they have opposite signs, {\displaystyle \ell (y)}\ell(y) increases linearly with y, and similarly if {\displaystyle |y|<1}{\displaystyle |y|<1}, even if it has the same sign (correct prediction, but not by enough margin).
What are
'''
nlp = spacy.load('en_core_web_lg')
doc = nlp(raw_text)
sentences = [sent.string.strip() for sent in doc.sents]
for sen in sentences:
    print('- %s' % sen)