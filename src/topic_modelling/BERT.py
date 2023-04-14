import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import umap.umap_ as UMAP
from hdbscan import HDBSCAN
import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
#If the following packages are not already downloaded, the following lines are needed 
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
import matplotlib.pyplot as plt

def BERT_model(data_frame, type_model = "train"):
    print("BERT_model")

    data = data_frame['Text']
    time = data_frame['Time']   
    data_list = data.to_list()
    time_list = time.to_list()

    '''
        Import all-MiniLM-L6-v2 pretrained model
        Then added prediction_data=True as a new parameter to HDBSCAN. 
        We need this to avoid an AttributeError when integrating our custom HDBSCAN step with BERTopic.
        Adding gen_min_span_tree adds another step to HDBSCAN that can improve the resultant clusters.
        We must also initialize a vectorizer_model to handle stopword removal during the c-TF-IDF step. 
        We will use the list of stopwords from NLTK.
        ensemble a few models to topic model the dataset

        Note 
        -----
        Bertopic may have dependecny conflicts that we took into account, however, it sometimes causes
        the model importation and tensorflow to be conflicted
        Thus we implemented a similar BERT transformer method to classify our data and topics accodingly
        
        Parameters
        ----------
        data : Takes in reviews.csv dataframe, data should have 3 columns -> sentiment, time, text.

        Returns
        -------
        topics split into top 10 words per topic
    '''

    try:
        print("training model")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

        print('die')
        umap_model = UMAP.UMAP(n_neighbors=3, n_components=3, min_dist=0.05)
        hdbscan_model = HDBSCAN(min_cluster_size=80, min_samples=40,
                            gen_min_span_tree=True,
                            prediction_data=True)
        print('loaded in')

        stopwords = list(stopwords.words('english')) + ['http', 'https', 'amp', 'com']

        # we add this to remove stopwords that can pollute topcs
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords)

        print('bert_topic')
        model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            top_n_words=10,
            language='english',
            calculate_probabilities=True,
            verbose=True
        )
        print("===== transforming bert =====")
        topics, probs = model.fit_transform(data)
        model.save("./bert_saved/nigel_bert")
        print("===== model run =====")

        print(model.visualize_barchart())
        print(model.get_topics())

        filtered_text = []
        lemmatizer = WordNetLemmatizer()

        for w in data:
            filtered_text.append(lemmatizer.lemmatize(w))

        # Step 2.5 - Create topic representation
        ctfidf_model = ClassTfidfTransformer()

        documents = pd.DataFrame({"Document": filtered_text,
                          "ID": range(len(filtered_text)),
                          "Topic": topics})
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = model._preprocess_text(documents_per_topic.Document.values)

        # Extract vectorizer and analyzer from BERTopic
        vectorizer = model.vectorizer_model
        analyzer = vectorizer.build_analyzer()

        # Extract features for Topic Coherence evaluation
        words = vectorizer.get_feature_names_out()
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topic_words = [[words for words, _ in model.get_topic(topic)] 
                    for topic in range(len(set(topics))-1)]

        # Evaluate
        coherence_model_c_v = CoherenceModel(topics=topic_words, 
                                        texts=tokens, 
                                        corpus=corpus,
                                        dictionary=dictionary, 
                                        coherence='c_v')
        coherence_c_v = coherence_model_c_v.get_coherence()
        print("===== model run =====")
        print("coherence score for model c_v")
        print(coherence_c_v)

        # Evaluate
        coherence_model_u_mass = CoherenceModel(topics=topic_words, 
                                        texts=tokens, 
                                        corpus=corpus,
                                        dictionary=dictionary, 
                                        coherence='u_mass')
        coherence_u_mass = coherence_model_u_mass.get_coherence()
        print("===== model run =====")
        print("coherence score for model u_mass")
        print(coherence_u_mass)

            #graphs:
        fig = model.visualize_barchart()
        print(fig)

        topics_over_time = model.topics_over_time(data_list, time_list, nr_bins=20)
        model.visualize_topics_over_time(topics_over_time, top_n_topics=20)

        topic_df = pd.DataFrame.from_dict(model.get_topics())
        return topic_df
        
    except:
        print("==== BERT TOPIC MODEL FAILED INITIALISATION ====")
        print("==== running manual bert")
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        embeddings = model.encode(data, show_progress_bar=True)

        umap_embeddings = UMAP.UMAP(n_neighbors=5, 
                            n_components=5, 
                            metric='cosine').fit_transform(embeddings)
        luster = HDBSCAN(min_cluster_size=5,
                          metric='euclidean',                      
                          cluster_selection_method='eom').fit(umap_embeddings)
        # Prepare data
        umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
        result = pd.DataFrame(umap_data, columns=['x', 'y'])
        result['labels'] = cluster.labels_

        # Visualize clusters
        fig, ax = plt.subplots(figsize=(20, 10))
        outliers = result.loc[result.labels == -1, :]
        clustered = result.loc[result.labels != -1, :]
        plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
        plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
        plt.colorbar()

        # cleaning the dataframe
        docs_df = pd.DataFrame(data)
        docs_df = docs_df.rename(columns={"Text": "Doc"})

        docs_df['Topic'] = cluster.labels_
        docs_df['Doc_ID'] = range(len(docs_df))
        docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})


        def c_tf_idf(documents, m, ngram_range=(1, 1)):
            count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
            t = count.transform(documents).toarray()
            w = t.sum(axis=1)
            tf = np.divide(t.T, w)
            sum_t = t.sum(axis=0)
            idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
            tf_idf = np.multiply(tf, idf)

            return tf_idf, count
        
        tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(data))

                
        def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
            words = count.get_feature_names()
            labels = list(docs_per_topic.Topic)
            tf_idf_transposed = tf_idf.T
            indices = tf_idf_transposed.argsort()[:, -n:]
            top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
            return top_n_words

        def extract_topic_sizes(df):
            topic_sizes = (df.groupby(['Topic'])
                            .Doc
                            .count()
                            .reset_index()
                            .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                            .sort_values("Size", ascending=False))
            return topic_sizes

        top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
        topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(11)
    
        for i in range(20):
            # Calculate cosine similarity
            similarities = cosine_similarity(tf_idf.T)
            np.fill_diagonal(similarities, 0)

            # Extract label to merge into and from where
            topic_sizes = docs_df.groupby(['Topic']).count().sort_values("Doc", ascending=False).reset_index()
            topic_to_merge = topic_sizes.iloc[-1].Topic
            topic_to_merge_into = np.argmax(similarities[topic_to_merge + 1]) - 1

            # Adjust topics
            docs_df.loc[docs_df.Topic == topic_to_merge, "Topic"] = topic_to_merge_into
            old_topics = docs_df.sort_values("Topic").Topic.unique()
            map_topics = {old_topic: index - 1 for index, old_topic in enumerate(old_topics)}
            docs_df.Topic = docs_df.Topic.map(map_topics)
            docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})

            # Calculate new topic words
            m = len(data)
            tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m)
            top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)

        topic_sizes = extract_topic_sizes(docs_df)
        print("==== classified topics====")
        print(topic_sizes.head(10))

        top_20_words_bert = pd.DataFrame.from_dict(top_n_words)
        return top_20_words_bert
    
    
    # if type_model == "test":
    #     # Load the serialized model
    #     model = BERTopic.load("./bert_saved/nigel_bert")
    #     print("===== model run =====")

    #     print(model.visualize_barchart())
    #     print(model.get_topics())

    #     filtered_text = []
    #     lemmatizer = WordNetLemmatizer()

    #     for w in data:
    #         filtered_text.append(lemmatizer.lemmatize(w))

    #     # Step 2.5 - Create topic representation
    #     ctfidf_model = ClassTfidfTransformer()

    #     documents = pd.DataFrame({"Document": filtered_text,
    #                       "ID": range(len(filtered_text)),
    #                       "Topic": topics})
    #     documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    #     cleaned_docs = model._preprocess_text(documents_per_topic.Document.values)

    #     # Extract vectorizer and analyzer from BERTopic
    #     vectorizer = model.vectorizer_model
    #     analyzer = vectorizer.build_analyzer()

    #     # Extract features for Topic Coherence evaluation
    #     words = vectorizer.get_feature_names_out()
    #     tokens = [analyzer(doc) for doc in cleaned_docs]
    #     dictionary = corpora.Dictionary(tokens)
    #     corpus = [dictionary.doc2bow(token) for token in tokens]
    #     topic_words = [[words for words, _ in model.get_topic(topic)] 
    #                 for topic in range(len(set(topics))-1)]

    #     # Evaluate
    #     coherence_model_c_v = CoherenceModel(topics=topic_words, 
    #                                     texts=tokens, 
    #                                     corpus=corpus,
    #                                     dictionary=dictionary, 
    #                                     coherence='c_v')
    #     coherence_c_v = coherence_model_c_v.get_coherence()
    #     print("===== model run =====")
    #     print("coherence score for model c_v")
    #     print(coherence_c_v)

    #     # Evaluate
    #     coherence_model_u_mass = CoherenceModel(topics=topic_words, 
    #                                     texts=tokens, 
    #                                     corpus=corpus,
    #                                     dictionary=dictionary, 
    #                                     coherence='u_mass')
    #     coherence_u_mass = coherence_model_u_mass.get_coherence()
    #     print("===== model run =====")
    #     print("coherence score for model u_mass")
    #     print(coherence_u_mass)

    # #graphs:
    # fig = model.visualize_barchart()
    # print(fig)

    # topics_over_time = model.topics_over_time(data_list, time_list, nr_bins=20)
    # model.visualize_topics_over_time(topics_over_time, top_n_topics=20)

    # topic_df = pd.DataFrame.from_dict(model.get_topics())
    # return topic_df
