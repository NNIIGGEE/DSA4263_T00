import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import gensim
from gensim.models import Phrases
#Prepare objects for LDA gensim implementation
from gensim import corpora
#Running LDA
from gensim import models
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models

""" --- POSITIVE --- """

def positive_LDA_topic_modelling(df_positive):

    tokens = []
    for sentence in df_positive['cleaned']:
        tokens.append(ast.literal_eval(sentence))

    # training a bi gram model in order to include those bigrams as tokens who occured at least 6 times
    # in the whole dataset
    bigram = gensim.models.Phrases(tokens, min_count=2, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    
    # including bigrams as tokens 
    sents = [ bigram_mod[token] for token in tokens]
    # Create Dictionary to keep track of vocab
    dct = corpora.Dictionary(tokens)

    # Create Corpus(Database)
    corpus = [dct.doc2bow(sent) for sent in sents]

    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    #randomstate = 12
    scores = []
    print("Coherence Score for Each Topic for Positive Sentiment")
    
    for k in range(3,10):
        # LDA model
        lda_model = gensim.models.LdaModel(corpus=corpus_tfidf, num_topics=k, 
                                                    id2word=dct, random_state=12)
        # to calculate score for coherence
        coherence_model_lda = CoherenceModel(model=lda_model, texts=sents, dictionary=dct, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print(k, coherence_lda)
        scores.append(coherence_lda)

    #chosen number of topic = 4 as the intertopic distance map shows a better result than 5 topics (topic 1 and 5 are relatively close tgt which means they're similar)
    import pyLDAvis.gensim_models

    selected_topics=4
    lda_model = gensim.models.LdaModel(corpus=corpus_tfidf, id2word=dct, num_topics=selected_topics,\
                                            random_state=12, chunksize=128, passes=10 )

    pyLDAvis.enable_notebook()
    results = pyLDAvis.gensim_models.prepare(lda_model, corpus_tfidf, dct, sort_topics=False)
    pyLDAvis.save_html(results, 'ldavis_english' +'.html')
    #results

    top_words_df = pd.DataFrame()
    for k in range(selected_topics):
        # top words with it's weight for a given id k 
        top_words = lda_model.show_topic(topicid=k)
        
        # only keep the word and discard the weight
        top_words_df['Topic {}'.format(k)] = [pair[0] for pair in top_words]
    print("Top Words for Each Positive Topics")
    print(top_words_df)
    
    predicted_topics = lda_model[corpus_tfidf]

    # Extract the predicted topic for each document
    predicted_topics = [max(prob, key=lambda x: x[1])[0] for prob in predicted_topics]


    # Append the predicted topics to the DataFrame
    df_positive['Topics'] = predicted_topics
    return df_positive


""" --- NEGATIVE --- """

def negative_LDA_topic_modelling(df_negative):
   

    tokens = []
    for sentence in df_negative['cleaned']:
        tokens.append(ast.literal_eval(sentence))

        # training a bi gram model in order to include those bigrams as tokens who occured at least 6 times
        # in the whole dataset
        bigram = gensim.models.Phrases(tokens, min_count=2, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        # including bigrams as tokens 
        sents = [ bigram_mod[token] for token in tokens]
        # Create Dictionary to keep track of vocab
        dct = corpora.Dictionary(tokens)


    print('Unique words before filtering/after pre-processing', len(dct))

    # Create Corpus(Database)
    corpus = [dct.doc2bow(sent) for sent in sents]

    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    #randomstate = 12
    scores = []
    print("Coherence Score for Each Topic for Negative Sentiment")

    for k in range(3,10):
        # LDA model
        lda_model = gensim.models.LdaModel(corpus=corpus_tfidf, num_topics=k, 
                                                    id2word=dct, random_state=12)
        # to calculate score for coherence
        coherence_model_lda = CoherenceModel(model=lda_model, texts=sents, dictionary=dct, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print(k, coherence_lda)
        scores.append(coherence_lda)

    selected_topics = np.argmax(scores)+3

    #chosen number of topic = 4 for negative sentiments
    import pyLDAvis.gensim_models

    selected_topics=4
    lda_model = gensim.models.LdaModel(corpus=corpus_tfidf, id2word=dct, num_topics=selected_topics,\
                                            random_state=12, chunksize=128, passes=10 )

    pyLDAvis.enable_notebook()
    results = pyLDAvis.gensim_models.prepare(lda_model, corpus_tfidf, dct, sort_topics=False)
    pyLDAvis.save_html(results, 'ldavis_english' +'.html')
    #results

    top_words_df = pd.DataFrame()
    for k in range(selected_topics):
        # top words with it's weight for a given id k 
        top_words = lda_model.show_topic(topicid=k)
        
        # only keep the word and discard the weight
        top_words_df['Topic {}'.format(k+4)] = [pair[0] for pair in top_words]
    
    print("Top Words for Each Negative Topics")
    print(top_words_df)

    predicted_topics = lda_model[corpus_tfidf]

    # Extract the predicted topic for each document
    predicted_topics = [(max(prob, key=lambda x: x[1])[0] + 4) for prob in predicted_topics]


    # Append the predicted topics to the DataFrame
    df_negative['Topics'] = predicted_topics

    return df_negative


def full_LDA_topic_modelling(dataframe):
    """
    Parameters:
    dataframe with predicted "Sentiment"
    
    Output:
    Predicted "Topics" column appended 
    """
    df = dataframe

    df_positive = df[df['Sentiment'] == 'positive']
    df_negative = df[df['Sentiment'] == 'negative']

    df_positive = positive_LDA_topic_modelling(df_positive)
    df_negative = negative_LDA_topic_modelling(df_negative)

    #combine the positive and negative dataset
    df_result = pd.concat([df_positive, df_negative])

    return df_result
   
