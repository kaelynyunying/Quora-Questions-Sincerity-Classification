import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import *
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.initializers import *
from keras.optimizers import *
import keras.backend as K
from keras.callbacks import *
import tensorflow as tf
import os
import time
import gc
import re
import random
from tqdm import tqdm
tqdm.pandas()
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.contrib.rnn import *
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
np.random.seed(2018)
import nltk
import pyLDAvis.gensim
from pprint import pprint
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

def preprocess(text):
    try:
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
        #print(result)
        return result
    except:
        return []
def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def format_topics_sentences(ldamodel, corpus):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    question = pd.Series(docs_sample['original'].tolist())
    target = pd.Series(docs_sample['target'].tolist())
    sent_topics_df = pd.concat([sent_topics_df, question, target], axis=1)
    return(sent_topics_df)

train = pd.read_csv('../input/train.csv')
print(str(len(train['question_text'])))

#pre-processing
docs_sample = train.sample(n = 130000, random_state = 1)
docs_sample['original'] = docs_sample['question_text']
docs_sample['question_text'] = docs_sample['question_text'].map(preprocess)

#building dictionary and corpus for LDA
dictionary = gensim.corpora.Dictionary(docs_sample['question_text'])
corpus = [dictionary.doc2bow(text) for text in docs_sample['question_text']]
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
print('Dictionary is built.')

#run LDA
#change num_topics if want to change the number of topics generated
lda_model = gensim.models.LdaModel(corpus, num_topics=30, id2word=dictionary, random_state=100,
                                           update_every=1,
                                           chunksize=10,
                                           passes=10,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

pyLDAvis.enable_notebook()

pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)

df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Question', 'Target']
df_dominant_topic.head(10)


# Display setting to show more characters in column
pd.options.display.max_colwidth = 100

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=False)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text", "Target"]

# Show
sent_topics_sorteddf_mallet.head(20)

insincere_example = sent_topics_sorteddf_mallet[(sent_topics_sorteddf_mallet['Target'] == 1) & (sent_topics_sorteddf_mallet['Topic_Num'] == 20.0)].copy()
insincere_example.head(20)

#Generate sincere/insincere question distribution for each topic
distribution = pd.DataFrame(sent_topics_sorteddf_mallet.groupby(['Topic_Num', 'Target']).size())
distribution.columns = ['Question Count']
distribution['propotion'] = distribution['Question Count'].divide(distribution['Question Count'] + distribution['Question Count'].shift(1))
print(distribution)

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything()

test = pd.read_csv("../input/test.csv")

train["question_text"] = train["question_text"].str.lower()
test["question_text"] = test["question_text"].str.lower()

def clean_tag(text):
    if '[math]' in text:
        text = re.sub('\[math\].*?math\]', '[formula]', text)
    if 'http' in text or 'www' in text:
        text = re.sub('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+', '[url]', text)
    return text
train["question_text"] = train["question_text"].apply(lambda x: clean_tag(x))
test["question_text"] = test["question_text"].apply(lambda x: clean_tag(x))
puncts=[',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', 
        '•', '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 
        '█', '…', '“', '★', '”', '–', '●', '►', '−', '¢', '¬', '░', '¡', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', 
        '—', '‹', '─', '▒', '：', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', '¯', '♦', '¤', '▲', '¸', '⋅', '‘', '∞', 
        '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '・', '╦', '╣', '╔', '╗', '▬', '❤', '≤', '‡', '√', '◄', '━', 
        '⇒', '▶', '≥', '╝', '♡', '◊', '。', '✈', '≡', '☺', '✔', '↵', '≈', '✓', '♣', '☎', '℃', '◦', '└', '‟', '～', '！', '○', 
        '◆', '№', '♠', '▌', '✿', '▸', '⁄', '□', '❖', '✦', '．', '÷', '｜', '┃', '／', '￥', '╠', '↩', '✭', '▐', '☼', '☻', '┐', 
        '├', '«', '∼', '┌', '℉', '☮', '฿', '≦', '♬', '✧', '〉', '－', '⌂', '✖', '･', '◕', '※', '‖', '◀', '‰', '\x97', '↺', 
        '∆', '┘', '┬', '╬', '،', '⌘', '⊂', '＞', '〈', '⎙', '？', '☠', '⇐', '▫', '∗', '∈', '≠', '♀', '♔', '˚', '℗', '┗', '＊', 
        '┼', '❀', '＆', '∩', '♂', '‿', '∑', '‣', '➜', '┛', '⇓', '☯', '⊖', '☀', '┳', '；', '∇', '⇑', '✰', '◇', '♯', '☞', '´', 
        '↔', '┏', '｡', '◘', '∂', '✌', '♭', '┣', '┴', '┓', '✨', '\xa0', '˜', '❥', '┫', '℠', '✒', '［', '∫', '\x93', '≧', '］', 
        '\x94', '∀', '♛', '\x96', '∨', '◎', '↻', '⇩', '＜', '≫', '✩', '✪', '♕', '؟', '₤', '☛', '╮', '␊', '＋', '┈', '％', 
        '╋', '▽', '⇨', '┻', '⊗', '￡', '।', '▂', '✯', '▇', '＿', '➤', '✞', '＝', '▷', '△', '◙', '▅', '✝', '∧', '␉', '☭', 
        '┊', '╯', '☾', '➔', '∴', '\x92', '▃', '↳', '＾', '׳', '➢', '╭', '➡', '＠', '⊙', '☢', '˝', '∏', '„', '∥', '❝', '☐', 
        '▆', '╱', '⋙', '๏', '☁', '⇔', '▔', '\x91', '➚', '◡', '╰', '\x85', '♢', '˙', '۞', '✘', '✮', '☑', '⋆', 'ⓘ', '❒', 
        '☣', '✉', '⌊', '➠', '∣', '❑', '◢', 'ⓒ', '\x80', '〒', '∕', '▮', '⦿', '✫', '✚', '⋯', '♩', '☂', '❞', '‗', '܂', '☜', 
        '‾', '✜', '╲', '∘', '⟩', '＼', '⟨', '·', '✗', '♚', '∅', 'ⓔ', '◣', '͡', '‛', '❦', '◠', '✄', '❄', '∃', '␣', '≪', '｢', 
        '≅', '◯', '☽', '∎', '｣', '❧', '̅', 'ⓐ', '↘', '⚓', '▣', '˘', '∪', '⇢', '✍', '⊥', '＃', '⎯', '↠', '۩', '☰', '◥', 
        '⊆', '✽', '⚡', '↪', '❁', '☹', '◼', '☃', '◤', '❏', 'ⓢ', '⊱', '➝', '̣', '✡', '∠', '｀', '▴', '┤', '∝', '♏', 'ⓐ', 
        '✎', ';', '␤', '＇', '❣', '✂', '✤', 'ⓞ', '☪', '✴', '⌒', '˛', '♒', '＄', '✶', '▻', 'ⓔ', '◌', '◈', '❚', '❂', '￦', 
        '◉', '╜', '̃', '✱', '╖', '❉', 'ⓡ', '↗', 'ⓣ', '♻', '➽', '׀', '✲', '✬', '☉', '▉', '≒', '☥', '⌐', '♨', '✕', 'ⓝ', 
        '⊰', '❘', '＂', '⇧', '̵', '➪', '▁', '▏', '⊃', 'ⓛ', '‚', '♰', '́', '✏', '⏑', '̶', 'ⓢ', '⩾', '￠', '❍', '≃', '⋰', '♋', 
        '､', '̂', '❋', '✳', 'ⓤ', '╤', '▕', '⌣', '✸', '℮', '⁺', '▨', '╨', 'ⓥ', '♈', '❃', '☝', '✻', '⊇', '≻', '♘', '♞', 
        '◂', '✟', '⌠', '✠', '☚', '✥', '❊', 'ⓒ', '⌈', '❅', 'ⓡ', '♧', 'ⓞ', '▭', '❱', 'ⓣ', '∟', '☕', '♺', '∵', '⍝', 'ⓑ', 
        '✵', '✣', '٭', '♆', 'ⓘ', '∶', '⚜', '◞', '்', '✹', '➥', '↕', '̳', '∷', '✋', '➧', '∋', '̿', 'ͧ', '┅', '⥤', '⬆', '⋱', 
        '☄', '↖', '⋮', '۔', '♌', 'ⓛ', '╕', '♓', '❯', '♍', '▋', '✺', '⭐', '✾', '♊', '➣', '▿', 'ⓑ', '♉', '⏠', '◾', '▹', 
        '⩽', '↦', '╥', '⍵', '⌋', '։', '➨', '∮', '⇥', 'ⓗ', 'ⓓ', '⁻', '⎝', '⌥', '⌉', '◔', '◑', '✼', '♎', '♐', '╪', '⊚', 
        '☒', '⇤', 'ⓜ', '⎠', '◐', '⚠', '╞', '◗', '⎕', 'ⓨ', '☟', 'ⓟ', '♟', '❈', '↬', 'ⓓ', '◻', '♮', '❙', '♤', '∉', '؛', 
        '⁂', 'ⓝ', '־', '♑', '╫', '╓', '╳', '⬅', '☔', '☸', '┄', '╧', '׃', '⎢', '❆', '⋄', '⚫', '̏', '☏', '➞', '͂', '␙', 
        'ⓤ', '◟', '̊', '⚐', '✙', '↙', '̾', '℘', '✷', '⍺', '❌', '⊢', '▵', '✅', 'ⓖ', '☨', '▰', '╡', 'ⓜ', '☤', '∽', '╘', 
        '˹', '↨', '♙', '⬇', '♱', '⌡', '⠀', '╛', '❕', '┉', 'ⓟ', '̀', '♖', 'ⓚ', '┆', '⎜', '◜', '⚾', '⤴', '✇', '╟', '⎛', 
        '☩', '➲', '➟', 'ⓥ', 'ⓗ', '⏝', '◃', '╢', '↯', '✆', '˃', '⍴', '❇', '⚽', '╒', '̸', '♜', '☓', '➳', '⇄', '☬', '⚑', 
        '✐', '⌃', '◅', '▢', '❐', '∊', '☈', '॥', '⎮', '▩', 'ு', '⊹', '‵', '␔', '☊', '➸', '̌', '☿', '⇉', '⊳', '╙', 'ⓦ', 
        '⇣', '｛', '̄', '↝', '⎟', '▍', '❗', '״', '΄', '▞', '◁', '⛄', '⇝', '⎪', '♁', '⇠', '☇', '✊', 'ி', '｝', '⭕', '➘', 
        '⁀', '☙', '❛', '❓', '⟲', '⇀', '≲', 'ⓕ', '⎥', '\u06dd', 'ͤ', '₋', '̱', '̎', '♝', '≳', '▙', '➭', '܀', 'ⓖ', '⇛', '▊', 
        '⇗', '̷', '⇱', '℅', 'ⓧ', '⚛', '̐', '̕', '⇌', '␀', '≌', 'ⓦ', '⊤', '̓', '☦', 'ⓕ', '▜', '➙', 'ⓨ', '⌨', '◮', '☷', 
        '◍', 'ⓚ', '≔', '⏩', '⍳', '℞', '┋', '˻', '▚', '≺', 'ْ', '▟', '➻', '̪', '⏪', '̉', '⎞', '┇', '⍟', '⇪', '▎', '⇦', '␝', 
        '⤷', '≖', '⟶', '♗', '̴', '♄', 'ͨ', '̈', '❜', '̡', '▛', '✁', '➩', 'ா', '˂', '↥', '⏎', '⎷', '̲', '➖', '↲', '⩵', '̗', '❢', 
        '≎', '⚔', '⇇', '̑', '⊿', '̖', '☍', '➹', '⥊', '⁁', '✢']

def clean_punct(x):
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, f' {punct} ')
    return x
train["question_text"] = train["question_text"].apply(lambda x: clean_punct(x))
test["question_text"] = test["question_text"].apply(lambda x: clean_punct(x))
## some config values 
embed_size = 300 # how big is each word vector
max_features = 200000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 72 # max number of words in a question to use #99.99%

## fill up the missing values
X = train["question_text"].fillna("_####_").values
X_test = test["question_text"].fillna("_####_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features, filters='')
tokenizer.fit_on_texts(list(X)+list(X_test))

X = tokenizer.texts_to_sequences(X)
X_test = tokenizer.texts_to_sequences(X_test)

## Pad the sentences 
X = pad_sequences(X, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

## Get the target values
Y = train['target'].values

sub = test[['qid']]
del train, test
gc.collect()

print("Finish preprocessing for train and test")

lem = WordNetLemmatizer()
word_index = tokenizer.word_index
max_features = len(word_index)+1

#Word embeddings
#Here the pre-trained word embeddings' dimension is set to 300. 
#We can use these embeddings by averaging or maxpooling and use it as unput for input layer
#The matrix = weight of the input layer.
def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) 
                            for o in open(EMBEDDING_FILE) 
                            if o.split(" ")[0] in word_index or o.split(" ")[0].lower() in word_index)

    emb_mean, emb_std = -0.005838499, 0.48782197
    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
        elif embeddings_index.get(word.capitalize()) is not None:
            embedding_matrix[i] = embeddings_index.get(word.capitalize())
        elif embeddings_index.get(word.upper()) is not None:
            embedding_matrix[i] = embeddings_index.get(word.upper())
    del embeddings_index
    gc.collect()        
    return embedding_matrix 

def load_para(word_index):
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) 
                            for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') 
                            if len(o)>100 and (o.split(" ")[0] in word_index or o.split(" ")[0].lower() in word_index))

    emb_mean, emb_std = -0.005838499, 0.48782197
    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
        elif embeddings_index.get(word.capitalize()) is not None:
            embedding_matrix[i] = embeddings_index.get(word.capitalize())
        elif embeddings_index.get(word.upper()) is not None:
            embedding_matrix[i] = embeddings_index.get(word.upper())
        
    del embeddings_index
    gc.collect()
    return embedding_matrix

seed_everything()
embedding_matrix_1 = load_glove(word_index)
embedding_matrix_3 = load_para(word_index)
embedding_matrix = np.mean((embedding_matrix_1, embedding_matrix_3), axis=0)
del embedding_matrix_1, embedding_matrix_3
gc.collect()
np.shape(embedding_matrix)

#Attention method 
class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.RandomUniform(seed=10000)
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None
class AdamW(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4,  # decoupled weight decay (1/4)
                 epsilon=1e-8, decay=0., **kwargs):
        super(AdamW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.wd = K.variable(weight_decay, name='weight_decay') # decoupled weight decay (2/4)
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        wd = self.wd # decoupled weight decay (3/4)

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) - lr * wd * p # decoupled weight decay (4/4)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'weight_decay': float(K.get_value(self.wd)),
                  'epsilon': self.epsilon}
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#First method uses LSTM and GRU
#The model is in 3 parts
#Layer 1 : Embedding Layer with 300 dimensions
#Layer 2: Hidden Layer (We use LSTM and GRU here)
#Layer 3 : Output Layer : dense layer to support activation method (either 'relu' or sigmoid)
def LSTM_GRU(spatialdropout=0.20, rnn_units=64, weight_decay=0.07):
    K.clear_session()       
    x_input = Input(shape=(maxlen,))
    
    emb = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False, name='Embedding')(x_input)
    emb = SpatialDropout1D(spatialdropout, seed=1024)(emb)

    rnn1 = Bidirectional(CuDNNLSTM(rnn_units, return_sequences=True, kernel_initializer=glorot_uniform(seed=111100), 
                           recurrent_initializer=Orthogonal(gain=1.0, seed=123000)))(emb)
    rnn2 = Bidirectional(CuDNNGRU(rnn_units, return_sequences=True, kernel_initializer=glorot_uniform(seed=111000), 
                           recurrent_initializer=Orthogonal(gain=1.0, seed=1203000)))(rnn1)

    x = concatenate([rnn1, rnn2])
    x = GlobalMaxPooling1D()(x)  
    x = Dense(32, activation='relu', kernel_initializer=glorot_uniform(seed=111000))(x)
    x = Dropout(0.2, seed=1024)(x)
    x_output = Dense(1, activation='sigmoid', kernel_initializer=glorot_uniform(seed=111100))(x)
    
    model = Model(inputs=x_input, outputs=x_output)
    model.compile(loss='binary_crossentropy', optimizer=AdamW(weight_decay=weight_decay),)
    return model

#Second method uses GRU
#The model is in 3 parts
#Layer 1 : Embedding Layer with 300 dimensions
#Layer 2: Hidden Layer (We use  GRU here)
#Layer 3 : Output Layer : dense layer to support activation method (either 'relu' or sigmoid)
def poolRNN(spatialdropout=0.2, gru_units=64, weight_decay=0.04):
    K.clear_session()
    inp = Input(shape=(maxlen,))
    embedding_layer = Embedding(max_features,
                                embed_size,
                                weights=[embedding_matrix],
                                input_length=maxlen,
                                trainable=False)(inp)
    embedding_layer = SpatialDropout1D(spatialdropout, seed=1024)(embedding_layer)

    rnn_1 = Bidirectional(CuDNNGRU(gru_units, return_sequences=True, 
                                   kernel_initializer=glorot_uniform(seed=10000), 
                                   recurrent_initializer=Orthogonal(gain=1.0, seed=123000)))(embedding_layer)

    last = Lambda(lambda t: t[:, -1], name='last')(rnn_1)
    maxpool = GlobalMaxPooling1D()(rnn_1)
    attn = AttentionWeightedAverage()(rnn_1)
    average = GlobalAveragePooling1D()(rnn_1)

    c = concatenate([last, maxpool, attn], axis=1)
    c = Reshape((3, -1))(c)
    c = Lambda(lambda x:K.sum(x, axis=1))(c)
    x = BatchNormalization()(c)
    x = Dense(32, activation='relu', kernel_initializer=glorot_uniform(seed=111000))(x)
    x = Dropout(0.2, seed=1024)(x)
    x = BatchNormalization()(x)
    output_layer = Dense(1, activation="sigmoid", kernel_initializer=glorot_uniform(seed=111000))(x)
    model = Model(inputs=inp, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer=AdamW(weight_decay=weight_decay))
    return model

#Third method uses LSTM  and CNN
#CNN -> Conv1D- CNN for 1 dimension which are texts.
#The model is in 3 parts
#Layer 1 : Embedding Layer with 300 dimensions
#Layer 2: Hidden Layer (We use LSTM and CNN here)
#Layer 3 : Output Layer : dense layer to support activation method (either 'relu' or sigmoid)
def BiLSTM_CNN(spatialdropout=0.2, rnn_units=64, filters=[100, 80, 30, 12], weight_decay=0.10):
    K.clear_session()       
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(rate=spatialdropout, seed=10000)(x)
    x = Bidirectional(CuDNNLSTM(rnn_units, return_sequences=True, 
                               kernel_initializer=glorot_uniform(seed=111000), 
                               recurrent_initializer=Orthogonal(gain=1.0, seed=123000)))(x)

    x1 = Conv1D(filters=filters[0], activation='relu', kernel_size=1, 
                padding='same', kernel_initializer=glorot_uniform(seed=110000))(x)
    x2 = Conv1D(filters=filters[1], activation='relu', kernel_size=2, 
                padding='same', kernel_initializer=glorot_uniform(seed=120000))(x)
    x3 = Conv1D(filters=filters[2], activation='relu', kernel_size=3, 
                padding='same', kernel_initializer=glorot_uniform(seed=130000))(x)
    x4 = Conv1D(filters=filters[3], activation='relu', kernel_size=5, 
                padding='same', kernel_initializer=glorot_uniform(seed=140000))(x)

    
    x1 = GlobalMaxPool1D()(x1)
    x2 = GlobalMaxPool1D()(x2)
    x3 = GlobalMaxPool1D()(x3)
    x4 = GlobalMaxPool1D()(x4)

    c = concatenate([x1, x2, x3, x4])
    x = Dense(32, activation='relu', kernel_initializer=glorot_uniform(seed=111000))(c)
    x = Dropout(0.2, seed=10000)(x)
    x = BatchNormalization()(x)
    x = Dense(1, activation="sigmoid", kernel_initializer=glorot_uniform(seed=110000))(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=AdamW(weight_decay=weight_decay))
    return model
#Here we do Stratified k validation 
#as well as ensemble method (Bagging)
#Print out submission.csv
def f1_smart(y_true, y_pred):
    args = np.argsort(y_pred)
    tp = y_true.sum()
    fs = (tp - np.cumsum(y_true[args[:-1]])) / np.arange(y_true.shape[0] + tp - 1, tp, -1)
    res_idx = np.argmax(fs)
    return 2 * fs[res_idx], (y_pred[args[res_idx]] + y_pred[args[res_idx + 1]]) / 2
kfold = StratifiedKFold(n_splits=7, random_state=10, shuffle=True)
bestscore = []
bestloss = []
y_test = np.zeros((X_test.shape[0], ))
oof = np.zeros((X.shape[0], ))
epochs = [8, 8, 7, 6]
val_list = []
for i, (train_index, valid_index) in enumerate(kfold.split(X, Y)):
    val_list += list(valid_index)
    print('FOLD%s'%(i+1))
    X_train, X_val, Y_train, Y_val = X[train_index], X[valid_index], Y[train_index], Y[valid_index]
    filepath="weights_best.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0, verbose=0)
    callbacks = [checkpoint, reduce_lr]
    if i == 0:
        model = LSTM_GRU(spatialdropout=0.20, rnn_units=64, weight_decay=0.07)
        print('LSTM_GRU(spatialdropout=0.20, rnn_units=64, weight_decay=0.07)')
    elif i == 1:
        model = poolRNN(spatialdropout=0.2, gru_units=128, weight_decay=0.04)
        print('poolRNN(spatialdropout=0.2, gru_units=128, weight_decay=0.04)')
    elif i == 2:
        model = BiLSTM_CNN(spatialdropout=0.2, rnn_units=128, filters=[100, 90, 30, 12], weight_decay=0.10)
        print('BiLSTM_CNN(spatialdropout=0.2, rnn_units=128, filters=[100, 90, 30, 12], weight_decay=0.10)')
    model.fit(X_train, Y_train, batch_size=512, epochs=epochs[i], 
              validation_data=(X_val, Y_val), verbose=0, callbacks=callbacks, 
              #class_weight={0:1, 1:1.25}
             )
    print("train logloss:%s"%model.history.history['loss'])
    print("val logloss:%s"%model.history.history['val_loss'])
    y_pred = model.predict([X_val], batch_size=1024, verbose=2)
    y_test += np.squeeze(model.predict([X_test], batch_size=1024, verbose=2))/3
    oof[valid_index] = np.squeeze(y_pred)
    f1, threshold = f1_smart(np.squeeze(Y_val), np.squeeze(y_pred))
    print('Optimal F1: {:.4f} at threshold: {:.4f}\n'.format(f1, threshold))
    bestscore.append(threshold)
    if i == 2:break


f1, threshold = f1_smart(np.squeeze(Y[val_list]), np.squeeze(oof[val_list]))
print('Optimal F1: {:.4f} at threshold: {:.4f}'.format(f1, threshold))
y_test = y_test.reshape((-1, 1)) 
pred_test_y = (y_test>threshold).astype(int) 
sub['prediction'] = pred_test_y 
sub.to_csv("submission.csv", index=False)