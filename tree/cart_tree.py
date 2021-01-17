from collections import Counter


# util classes
class QuestionNode:
    def __init__(self, question, left=None, right=None):
        self.left = left
        self.right = right
        self.question = question

    def __str__(self):
        return str(self.question)

    __repr__ = __str__


class LabelNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def __str__(self):
        return str(self.value)

    __repr__ = __str__


class ContinuousQuestion:
    def __init__(self, column_idx, value):
        if not isinstance(value, float):
            raise Exception('Value is not continuous')
        self.value = value
        self.column_idx = column_idx

    def __str__(self):
        return 'f{} < {}'.format(self.column_idx, self.value)

    def __repr__(self):
        return 'ContinuousQuestion({}, {})'.format(self.column_idx, self.value)

    def checkAnswer(self, ans):
        return ans < self.value


class DiscreteQuestion:
    def __init__(self, column_idx, value):
        if not isinstance(value, int) and not isinstance(value, str):
            raise Exception('Value is not discrete')
        self.value = value
        self.column_idx = column_idx

    def __str__(self):
        return 'f{} = {}'.format(self.column_idx, self.value)

    def __repr__(self):
        return 'DiscreteQuestion({}, {})'.format(self.column_idx, self.value)

    def checkAnswer(self, ans):
        return ans == self.value


# util function
def ginni(labels):
    """
    Calculate ginni index. More info about it in wiki
    :param labels: List<Integer> - label column in the dataset
    :return: float (range from 0 to 1) - ginni index corresponding to the dataset
    """
    value2freq = Counter(labels)
    g = 0
    for unique_value in value2freq:
        g += (value2freq[unique_value] / len(labels)) ** len(value2freq)
    return 1 - g


def getQuestion(column_idx, value):
    if not isinstance(value, int) and not isinstance(value, str) and not isinstance(value, float):
        raise ValueError('Value should be string, integer, or float')
    if isinstance(value, int) or isinstance(value, str):
        return DiscreteQuestion(column_idx, value)
    return ContinuousQuestion(column_idx, value)


def generateAllPossibleQuestion(feature_matrix):
    question = []
    if len(feature_matrix) > 0:
        for i in range(len(feature_matrix)):
            for j in range(len(feature_matrix[i])):
                question.append(getQuestion(j, feature_matrix[i][j]))
    return question


def partition_data(question, feature_matrix, label_vector):
    data = zip(feature_matrix, label_vector)
    true_instance, false_instance = [], []
    for feature_vector, label in data:
        if question.checkAnswer(feature_vector[question.column_idx]):
            true_instance.append((feature_vector, label))
        else:
            false_instance.append((feature_vector, label))
    return true_instance, false_instance


def find_best_question(feature_matrix, label_vector):
    questions = generateAllPossibleQuestion(feature_matrix)

    # calculate parent ginni
    parent_ginni = ginni(label_vector)

    # loops through all possible questions, find the best question
    max_information_gain = None
    best_question = None
    best_split = None
    for question in questions:
        # partition the data based on the question
        true_data, false_data = partition_data(question, feature_matrix, label_vector)

        # calculate each node ginni
        true_node_ginni = ginni(list(map(lambda x: x[1], true_data)))
        false_node_ginni = ginni(list(map(lambda x: x[1], false_data)))

        # calculate average child ginni
        child_ginni = len(true_data) / len(feature_matrix) * true_node_ginni + len(false_data) / len(
            feature_matrix) * false_node_ginni

        # calculate information gain = parent ginni - child ginni
        information_gain = parent_ginni - child_ginni

        if max_information_gain is None or max_information_gain < information_gain:
            max_information_gain = information_gain
            best_question = question
            best_split = (true_data, false_data)

    return best_question, best_split, max_information_gain


class CARTDecisionTreeClassifier:
    def __init__(self):
        self.root = None

    def _build_tree(self, feature_matrix, label_vector):
        best_question, (true_data, false_data), max_information_gain = find_best_question(feature_matrix, label_vector)

        if max_information_gain == 0:
            return LabelNode(label_vector[0])

        root = QuestionNode(best_question)

        # decouple feature matrix and label vector
        true_feature_mat, true_label_vec = list(map(lambda x: x[0], true_data)), list(map(lambda x: x[1], true_data))
        false_feature_mat, false_label_vec = list(map(lambda x: x[0], false_data)), list(
            map(lambda x: x[1], false_data))

        root.left = self._build_tree(false_feature_mat, false_label_vec)
        root.right = self._build_tree(true_feature_mat, true_label_vec)

        return root

    def fit(self, feature_matrix, label_vector):
        self.root = self._build_tree(feature_matrix, label_vector)

    def _predict_feature(self, feature_vector):
        current_node = self.root
        while not isinstance(current_node, LabelNode):
            if current_node.question.checkAnswer(feature_vector[current_node.question.column_idx]):
                current_node = current_node.right
            else:
                current_node = current_node.left

        return current_node.value

    def predict(self, feature_matrix):
        predictions = []
        for feature_vector in feature_matrix:
            predictions.append(self._predict_feature(feature_vector))

        return predictions
