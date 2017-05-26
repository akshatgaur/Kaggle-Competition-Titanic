from mrjob.job import MRJob
import math
import subprocess
import random


class MRRandomForest(MRJob):

    def mapper(self, _, filepath):
        filepath = str(filepath)
        train_path, test_path = filepath.split(" ")
        train_file = subprocess.Popen(["hadoop", "fs", "-cat", train_path], stdout=subprocess.PIPE)
        #train_file = open(train_path)
        train_data = []

        # format train and test file as per type of feature

        for line in train_file.stdout:
        # for line in train_file:
            words = line.split(";")
            temp = []
            # leave label
            for i in words[:-1]:
                try:
                    temp.append(float(i))
                except:
                    temp.append(str(i))

            #add label as string and remove '\n' from the end
            temp.append(str(words[-1][:-1]))
            train_data.append(temp)

        test_file = subprocess.Popen(["hadoop", "fs", "-cat", test_path], stdout=subprocess.PIPE)
        # test_file = open(test_path)
        test_data = []
        # passenger id in test data
        id = []
        for line in test_file.stdout:
        # for line in test_file:
            words = line.split(";")
            id.append(words[0])
            temp = []
            for i in words[1:]:
                try:
                    temp.append(float(i))
                except:
                    temp.append(str(i))

            test_data.append(temp)

        # create tree based on the train set with random samples
        selected_features = random.sample(range(len(train_data[0]) - 1), 6)
        obj = TreeNode()
        tree = obj.create_tree(train_data, selected_features)

        #print the tree
        # res = []
        # obj.display(tree, " ", res)
        # f = open(train_path.replace(".csv", "") + '.txt', 'w')
        # f.write("\n".join(res))

        # predict the labels for test data
        for i, sample in enumerate(test_data):
            dic = obj.predict(tree, sample)
            for key in dic:
                # pass these values to reducer
                yield (int(id[i]), int(key))

    def reducer(self, id, counts):
        dic = {}
        for val in counts:
            if val not in dic:
                dic[val] = 0
            dic[val] += 1
        values = list(dic.values())
        keys = list(dic.keys())

        yield (id, keys[values.index(max(values))])


class TreeNode:

    def __init__(self, col=-1, entropy=0, value=None, label_set=None, true_node=None, false_node=None, labels=None):
        self.col = col
        self.value = value
        self.true_node = true_node
        self.false_node = false_node
        self.labels = labels
        self.entropy = entropy
        self.label_set = label_set

    def create_tree(self, data, selected_features):
        if len(data) == 0:
            return TreeNode()
        max_gain = 0.0
        final_lists = None
        branching_feat_val = None
        # feature_count = len(data[0]) - 1

        divideSet_obj = DivideSet()
        data_score = divideSet_obj.entropy(data)

        # check gain for each feature for each set of value
        # choose selected features only
        #for col in range(feature_count):
        for col in selected_features:
            col_val = []

            for i, row in enumerate(data):
                if row[col] not in col_val:
                    col_val.append(row[col])

            # divide data for each value of that row
            for val in col_val:
                list1, list2 = divideSet_obj.divide(data, col, val)

                # cal information gain
                prob = len(list1)/float(len(data))
                info_gain = (data_score - prob * divideSet_obj.entropy(list1) - (1 - prob) *divideSet_obj.entropy(list2))

                # positive = 0
                # negative = 0
                if info_gain > max_gain and (len(list1) > 0 and len(list2) > 0):
                    max_gain = info_gain
                    final_lists = (list1, list2)
                    branching_feat_val = (col, val)

            # sub tree creation
        if max_gain > 0:
            true_node = self.create_tree(final_lists[0], selected_features)
            false_node = self.create_tree(final_lists[1], selected_features)
            return TreeNode(col=branching_feat_val[0],
                            label_set=divideSet_obj.label_count(final_lists[0] + final_lists[1]),
                            entropy=divideSet_obj.entropy(final_lists[0] + final_lists[1]),
                            value=branching_feat_val[1], true_node=true_node, false_node=false_node)


        else:
            return TreeNode(labels=divideSet_obj.label_count(data))

    def predict(self, tree, test):

        if tree.labels != None:
            return tree.labels

        val = test[tree.col]
        node = None

        if isinstance(val, int) or isinstance(val, float):
            if val >= tree.value:
                node = tree.true_node
            else:
                node = tree.false_node
        else:
            if val == tree.value:
                node = tree.true_node
            else:
                node = tree.false_node

        return self.predict(node, test)

    def display(self, tree, indent, res):

        if tree.labels != None:
            res.append(indent + "entropy: " + str(tree.entropy))
            res.append(indent + str(tree.labels))
            return

        res.append(indent +"entropy: "+ str(tree.entropy))
        res.append(indent + str(tree.label_set))

        if tree.true_node:
            res.append( indent + str(self.col_name(tree.col)) + " >= " + str(tree.value))
            self.display(tree.true_node, indent + '| ', res)
        if tree.false_node:
            res.append( indent + str(self.col_name(tree.col)) + " < " + str(tree.value))
            self.display(tree.false_node, indent + '| ', res)

    def col_name(self, col):

        if col == 0:
            return 'Age'
        if col == 1:
            return 'Embarked=C'
        if col == 2:
            return 'Embarked=Q'
        if col == 3:
            return 'Embarked=S'
        if col == 4:
            return 'Fare'
        if col == 5:
            return 'Parch'
        if col == 6:
            return 'Sex=female'
        if col == 7:
            return 'Sex=male'
        if col == 8:
            return 'SibSp'
        if col == 9:
            return 'Name'
        if col == 10:
            return 'Cabin'
        if col == 11:
            return 'Pclass=1'
        if col == 12:
            return 'Pclass=2'
        if col == 13:
            return 'Pclass=3'


class DivideSet:

    def divide(self, data, c, value):

        if isinstance(value, int) or isinstance(value, float):
            list1 = [data[i] for i in range(len(data)) if data[i][c] >= value]
            list2 = [data[i] for i in range(len(data)) if data[i][c] < value]
        else:
            list1 = [data[i] for i in range(len(data)) if data[i][c] == value]
            list2 = [data[i] for i in range(len(data)) if data[i][c] != value]

        return list1, list2

    def label_count(self, data):
        labels = {}
        for r in data:

            label = r[-1]
            if label not in labels:
                labels[label] = 0
            labels[label] += 1
        return labels

    def entropy(self, data):

        ent = 0.0
        labels = self.label_count(data)
        for label in labels.keys():
            prob = labels[label] / float(len(data))
            ent -= (prob * math.log(prob) / math.log(2))

        return ent


if __name__ == '__main__':
    MRRandomForest.run()



