from abc import ABC, abstractmethod


class AbstractDecisionTree(ABC):
    @abstractmethod
    def _split(self, data):
        """function to split data. 
        Decision tree usually operate by splitting data based on some criteria
        This function should be capable of doing this

        Args:
            data (array-like): shape (m, n+1)
            m - number of training examples
            n - number of fearures
            This arg should be the concatenation of label vector into feature matrix

        Returns:
            a list of partition nodes
        """
        pass

    @abstractmethod
    def fit(self, x, y, **kwargs):
        """function to build decision tree on data set.
        Warning: this function will change internal behaviour of the object.

        Args:
            x (array-like): shape (m, n)
            m - number of training examples
            n - number of features
            this is feature matrix

            y (list or array-like): shape (m,)
            m - number of training examples
            this is label vector.
        """
        pass

    @abstractmethod
    def _predictEach(self, x):
        """predict a single example

        Args:
            x (array-like): shape (1, n)
            n - number of features. 
            Should be equal to number of features of x passing through fit function

        Returns:
            prediction (array-like): shape (1,)
            predicted label of x
        """
        pass

    @abstractmethod
    def predict(self, x):
        """predict new examples

        Args:
            x (array-like): shape (*, n)
            n - number of features.
            This function should operate on with any number of examples

        Returns:
            predictions (array-like): shape (*,). 
            The length of prediction is equal to the number of x's rows.
        """
        pass
