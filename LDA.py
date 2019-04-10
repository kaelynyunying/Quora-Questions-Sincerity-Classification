import pandas as pd
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

#method for lemmatization and stemming
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

train = pd.read_csv('data/train.csv')
print(str(len(train['question_text'])))

#pre-processing
docs_sample = train.sample(n = 130000, random_state = 1)
docs_sample['original'] = docs_sample['question_text']
docs_sample['question_text'] = docs_sample['question_text'].map(preprocess)
print('Pre-processing is done.')

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







