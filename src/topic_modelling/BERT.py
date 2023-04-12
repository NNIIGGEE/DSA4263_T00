import pandas as pd
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


def BERT_model(data, type_model = "test"):

    data = data['Text']
    time = data['Time']   
    data_list = data.to_list()
    time_list = time.to_list()

    '''
        You might notice that we have added prediction_data=True as a new parameter to HDBSCAN. We need this to avoid an AttributeError when integrating our custom HDBSCAN step with BERTopic. Adding gen_min_span_tree adds another step to HDBSCAN that can improve the resultant clusters.
        We must also initialize a vectorizer_model to handle stopword removal during the c-TF-IDF step. We will use the list of stopwords from NLTK but add a few more tokens that seem to pollute the results.

        Parameters
        ----------
        data : Takes in cleaned dataframe, data should have 5 columns -> label, time, cleaned, text and cleaned2

        Returns
        -------
        Save trained model
    '''

    if type_model == "train":

        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        umap_model = UMAP.UMAP(n_neighbors=3, n_components=3, min_dist=0.05)
        hdbscan_model = HDBSCAN(min_cluster_size=80, min_samples=40,
                            gen_min_span_tree=True,
                            prediction_data=True)
        

        stopwords = list(stopwords.words('english')) + ['http', 'https', 'amp', 'com']

        # we add this to remove stopwords that can pollute topcs
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords)

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

        # Save the model in the previously created folder with the name 'my_best_model'
        
    if type_model == "test":
        # Load the serialized model
        model = BERTopic.load("./bert_saved/nigel_bert")
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




