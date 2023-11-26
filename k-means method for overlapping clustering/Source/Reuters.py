from bs4 import BeautifulSoup
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

class Reuters():

    def __init__(self, data, f):
        self.data = data
        self.f = f

    def get_Reuters(self, articles, tags):

        """
        Given a list of articles and their corresponding tags, this function creates three subsets of articles from the Reuters-21578 dataset based on their tags.
        The subsets are as follows:
        - Reuters-1: contains the 1156 articles having at least one tag among the the set of 10 categories {gold, ipi, ship, yen, dlr, money-fx, acq, rice, grain, and crude}.
        - Reuters-2: contains the 1308 articles having at least one tag among the the set of 10 categories {coffee, sugar, trade, rubber, earn, cpi, cotton, alum, bop, and jobs}.
        - Reuters-3: contains the 333 articles having at least one tag among the the set of 10 categories {gnp, interest, veg-oil, oilseed, corn, nat-gas, carcass, livestock, wheat, and soybean}.

        Input:
        - articles (list): A list of articles represented as strings.
        - tags (list): A list of lists where each sublist contains the tags associated with a corresponding article in the articles list.

        Output:
        - reuters_1_articles (list): A list of articles belonging to Reuters-1 subset.
        - reuters_2_articles (list): A list of articles belonging to Reuters-2 subset.
        - reuters_3_articles (list): A list of articles belonging to Reuters-3 subset.
        """

        # Reuters tags
        reuters_1_necessary_tags = {'gold', 'ipi', 'ship', 'yen', 'dlr', 'money-fx', 'acq', 'rice', 'grain', 'crude'}
        reuters_2_necessary_tags = {'coffee', 'sugar', 'trade', 'rubber', 'earn', 'cpi', 'cotton', 'alum', 'bop', 'jobs'}
        reuters_3_necessary_tags = {'gnp', 'interest', 'veg-oil', 'oilseed', 'corn', 'nat-gas', 'carcass', 'livestock', 'wheat', 'soybean'}

        reuters_1_articles, reuters_2_articles, reuters_3_articles = [], [], []
        reuters_1_tags, reuters_2_tags, reuters_3_tags = [], [], []

        for reuters_necessary_tags, reuters_articles, numb_articles, reuters_tags in zip(
                (reuters_1_necessary_tags, reuters_2_necessary_tags, reuters_3_necessary_tags),
                (reuters_1_articles, reuters_2_articles, reuters_3_articles),
                (1156, 1308, 333),
                (reuters_1_tags, reuters_2_tags, reuters_3_tags)):
            for i in range(len(articles)):
                if len(reuters_articles) > numb_articles: break
                tag_list = []
                for tag in tags[i]:
                    if tag in reuters_necessary_tags: tag_list.append(tag)
                if len(tag_list) > 0:
                    reuters_articles.append(articles[i])
                    reuters_tags.append(tag_list)

        print(f"Reuters-1 contains {len(reuters_1_articles)} articles.")
        print(f"Reuters-2 contains {len(reuters_2_articles)} articles.")
        print(f"Reuters-3 contains {len(reuters_3_articles)} articles.")
        print(f"Reuters-1 contains {len(reuters_1_tags)} tags.")
        print(f"Reuters-2 contains {len(reuters_2_tags)} tags.")
        print(f"Reuters-3 contains {len(reuters_3_tags)} tags.")

        return reuters_1_articles, reuters_2_articles, reuters_3_articles, reuters_1_tags, reuters_2_tags, reuters_3_tags

    def get_datasets(self):

        """
        Reads the datafiles of Reuters-21578 dataset and obtains the articles and tags associated.

        Output:
        - reuters_1_articles: List of articles belonging to Reuters-1 subset.
        - reuters_2_articles: List of articles belonging to Reuters-2 subset.
        - reuters_3_articles: List of articles belonging to Reuters-3 subset.
        """

        # create empty lists to store the articles and tags
        articles = []
        tags = []

        # loop over all the SGML files in the range
        for i in range(22):
            # generate the file name
            file_name = f"/reut2-{str(i).zfill(3)}.sgm"
            file_name = self.data + file_name

            # read the file contents
            with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
                file_contents = f.read()

            # parse the SGML using BeautifulSoup
            soup = BeautifulSoup(file_contents, 'html.parser')

            # extract the text content and tags of each article
            for article in soup.find_all('reuters'):
                # extract the article text
                text = article.find('text').text.strip()

                # add the article text to the list of articles
                articles.append(text)

                # extract the tags for the article
                topic_tags = article.topics
                if topic_tags is not None:
                    tags.append([tag.text for tag in topic_tags.find_all('d')])
                else:
                    tags.append([])

        # print the number of articles and tags processed
        print(f"Articles and Tags uploaded successfully.")
        print(f"Processed {len(articles)} articles and {len(tags)} tag sets.")

        # call the get_Reuters function to split articles into the three subsets
        return self.get_Reuters(articles, tags)

    def extract_features(self, articles_list):

        """
        Extract a set of features from a list of articles.

        Input:
        articles_list (list): A list of strings, where each string represents an article.

        Output:
        selected_tokens (list): A list of strings, where each string represents a selected token.
        """

        # Create a dictionary to store the count of each token across all articles
        token_count = {}

        # Loop over all articles in the list and count the occurrence of each token
        for article in articles_list:
            # Tokenize the article text
            tokens = article.split()

            # Update the count for each token
            for token in set(tokens):
                if token in token_count:
                    token_count[token] += 1
                else:
                    token_count[token] = 1

        # Extract the tokens that occur in at least three articles
        selected_tokens = [token for token, count in token_count.items() if count >= 3]

        # Return the selected tokens as the feature set
        return selected_tokens

    def create_td_matrix(self, articles_list, features):

        """
        Converts a list of articles into a term-document matrix using the selected features.

        Input:
        - articles_list (list): A list of articles where each article is represented as a string.
        - features (list): A list of selected features (i.e., tokens) to include in the term-document matrix.

        Output:
        - td_matrix (scipy.sparse.csr_matrix): A sparse term-document matrix of shape (num_articles, num_features) where each element represents the count of a feature in an article.
        """

        # Create a CountVectorizer object with the selected features
        vectorizer = CountVectorizer(vocabulary=features)

        # Convert the articles list to a document-term matrix
        td_matrix = vectorizer.fit_transform(articles_list)

        return td_matrix

    def lsa_reparameterization(self, td_matrix, num_topics):

        """
        Performs Latent Semantic Analysis (LSA) reparameterization on a given term-document matrix to provide a semantic indexing for each document.

        Input:
        - td_matrix (numpy.ndarray): A term-document matrix of shape (num_terms, num_documents) where each element represents the term frequency-inverse document frequency (TF-IDF) weight of a term in a document.
        - num_topics (int): The number of topics to extract from the td_matrix using truncated singular value decomposition (SVD).

        Output:
        - topic_matrix (numpy.ndarray): A matrix of shape (num_topics, num_terms) where each row represents a topic and each element represents the weight of a term in that topic.
        - normalized_document_matrix (numpy.ndarray): A matrix of shape (num_documents, num_topics) where each row represents a document and each element represents the weight of a topic in that document. The rows are normalized to have unit length.

        """

        # Perform truncated SVD on the TD matrix to extract the top num_topics topics
        svd = TruncatedSVD(n_components=num_topics)
        svd.fit(td_matrix)
        topic_matrix = svd.components_
        document_matrix = svd.transform(td_matrix)

        # Normalize the document matrix to have unit length
        document_norms = np.linalg.norm(document_matrix, axis=1, keepdims=True)
        normalized_document_matrix = document_matrix / document_norms

        return topic_matrix, normalized_document_matrix

    def preprocess_reuters(self):

        """
        Feature extraction of the subsets Reuters-1, Reuters-2 and Reuters-3.

        Output:
        reuters_1_lsa[1]: a list of normalized document matrix from LSA process belonging to Reuters-1 subset.
        reuters_2_lsa[1]: a list of normalized document matrix from LSA process belonging to Reuters-2 subset.
        reuters_3_lsa[1]: a list of normalized document matrix from LSA process belonging to Reuters-3 subset.
        reuters_1_tags: a list of tags corresponding to articles in the Reuters-1 subset.
        reuters_1_tags: a list of tags corresponding to articles in the Reuters-2 subset.
        reuters_1_tags: a list of tags corresponding to articles in the Reuters-3 subset.
        """

        reuters_1_articles, reuters_2_articles, reuters_3_articles, reuters_1_tags, reuters_2_tags, reuters_3_tags = self.get_datasets()
        # Get the features
        reuters_1_features, reuters_2_features, reuters_3_features = self.extract_features(reuters_1_articles), self.extract_features(reuters_2_articles), self.extract_features(reuters_3_articles)
        # Get the document-term matrix
        reuters_1_td_matrix, reuters_2_td_matrix, reuters_3_td_matrix = self.create_td_matrix(reuters_1_articles, reuters_1_features), self.create_td_matrix(reuters_2_articles, reuters_2_features), self.create_td_matrix(reuters_3_articles, reuters_3_features)

        for num, reuters_features, reuters_td_matrix in zip(range(1, 4),
                                                            (reuters_1_features, reuters_2_features, reuters_3_features),
                                                            (reuters_1_td_matrix, reuters_2_td_matrix, reuters_3_td_matrix)):
            print(f"Number of features extracted from Reuters-{num}: {len(reuters_features)}")
            print(f"Shape of td_matrix for Reuters-{num}: {reuters_td_matrix.shape}\n")

        # Get the matrices from the lsa process
        reuters_1_lsa_lim = 10
        reuters_2_lsa_lim = 10
        reuters_3_lsa_lim = 10
        reuters_1_lsa, reuters_2_lsa, reuters_3_lsa = self.lsa_reparameterization(reuters_1_td_matrix, reuters_1_lsa_lim), self.lsa_reparameterization(reuters_2_td_matrix, reuters_2_lsa_lim), self.lsa_reparameterization(reuters_3_td_matrix, reuters_3_lsa_lim)
        for num, reuters_lsa in zip(range(1, 4), (reuters_1_lsa, reuters_2_lsa, reuters_3_lsa)):
            print(f"Shape of the topic matrix from LSA process for Reuters-{num}: {reuters_lsa[0].shape}")
            print(f"Shape of the normalized document matrix from LSA process for Reuters-{num}: {reuters_lsa[1].shape}\n")
        return reuters_1_lsa[1], reuters_2_lsa[1], reuters_3_lsa[1], reuters_1_tags, reuters_2_tags, reuters_3_tags