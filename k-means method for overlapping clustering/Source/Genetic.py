import pandas as pd
from scipy.io import arff
from sklearn.impute import KNNImputer

class Genetic():

    def __init__(self, f):
        self.f = f

    def eraseClassColumn(self, df):
        """
        Removes the class column from the given DataFrame and returns a new DataFrame without it, along with a list of
        the labels for each instance.

        Args:
        - df (DataFrame): The DataFrame to remove the class column from.

        Returns:
        - dfaux (DataFrame): A new DataFrame without the class column.
        - labels (list): A list of the labels for each instance.
        """
        dfaux = df.iloc[:, :-14]
        labels_aux = df.iloc[:, -14:].values.tolist()
        labels = []
        for instance in labels_aux:
            instance_label = []
            for id in range(len(instance)):
                if instance[id] == '0': continue
                elif instance[id] == '1': instance_label.append('Class'+str(id))
            labels.append(instance_label)
        return dfaux, labels

    def applyOneHotEncoding(self, df):
        """
        Applies one-hot encoding to the categorical columns in the given DataFrame and returns a new DataFrame with the
        encoded columns.

        Args:
        - df (DataFrame): The DataFrame to apply one-hot encoding to.

        Returns:
        - df (DataFrame): A new DataFrame with the encoded columns.
        """
        categorical = []
        for col in df.columns:
            if df[col].dtype == object:
                categorical.append(col)
        df = pd.get_dummies(df, columns=categorical)
        return df

    def normalizeDataset(self, df):
        """
        Normalizes the values in the given DataFrame to be between 0 and 1 and returns a new DataFrame with the
        normalized values.

        Args:
        - df (DataFrame): The DataFrame to normalize.

        Returns:
        - df_norm (DataFrame): A new DataFrame with the normalized values.
        """
        df_norm = df.copy()
        for col in df_norm.columns:
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
        return df_norm

    def replaceNumericalMissings(self, df):
        """
        Replaces missing values in the numerical columns of the given DataFrame using KNN imputation and returns a new
        DataFrame with the imputed values.

        Args:
        - df (DataFrame): The DataFrame to replace missing values in.

        Returns:
        - df_copy (DataFrame): A new DataFrame with the imputed values.
        """
        numerical = []
        df_copy = df.copy()
        for ind, col in enumerate(df.columns.values):
            if df[col].dtype != object:
                numerical.append(ind)
        if len(numerical) == 0:
            return df_copy
        dd = df.iloc[:, numerical]
        colnames = dd.columns
        imputer = KNNImputer(weights='distance')
        imputer.fit(dd)
        ddarray = imputer.transform(dd)
        ddclean = pd.DataFrame(ddarray, columns=colnames)
        for col in ddclean.columns:
            df_copy[col] = ddclean[col]
        return df_copy

    def preprocessDataset(self, filename):
        """
        Preprocesses a dataset in the ARFF format by performing the following steps:
        1. Load the dataset from the ARFF file using the scipy.io.arff.loadarff function.
        2. Convert the resulting NumPy array to a Pandas dataframe.
        3. Convert any byte strings in the dataframe to regular strings.
        4. Erase the class column from the dataframe and save it as a separate list of labels.
        5. Replace any missing numerical values in the dataframe using the KNNImputer from scikit-learn.
        6. Apply one-hot encoding to any categorical columns in the dataframe using Pandas get_dummies function.
        7. Normalize all the numerical columns in the dataframe to the range [0, 1].
        8. Return the preprocessed data as a NumPy array and the labels as a list of lists.

        Parameters:
        - filename: A string specifying the path to the ARFF file containing the dataset.

        Returns:
        - X: A NumPy array of shape (n_samples, n_features) containing the preprocessed data.
        - labels: A list of lists, where each sublist contains the names of the classes that correspond to
        the labels for the corresponding row in the preprocessed data. The length of this list should be equal
        to the number of samples in the preprocessed data.
        """
        data = arff.loadarff(filename)
        df = pd.DataFrame(data[0])
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.decode('utf-8')

        df, labels = self.eraseClassColumn(df)
        df = self.replaceNumericalMissings(df)
        df = self.applyOneHotEncoding(df)
        df = self.normalizeDataset(df)
        return df.values, labels