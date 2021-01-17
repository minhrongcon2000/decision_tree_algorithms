import unittest
from tree.cart_tree import CARTDecisionTreeClassifier
from sklearn.datasets import load_iris


class MyTestCase(unittest.TestCase):
    def test_binary(self):
        feature_matrix = [[1, 1], [0, 1], [0, 0], [1, 0]]
        label_vector = [1, 0, 0, 0]

        tree = CARTDecisionTreeClassifier()
        tree.fit(feature_matrix, label_vector)
        predictions = tree.predict(feature_matrix)
        self.assertEqual(predictions, label_vector)

    def test_iris(self):
        iris_data = load_iris()
        tree = CARTDecisionTreeClassifier()
        tree.fit(iris_data.data, iris_data.target)
        self.assertEqual(tree.predict(iris_data.data), list(iris_data.target))


if __name__ == '__main__':
    unittest.main()
