from mrjob.job import MRJob


class MRKNN(MRJob):
    def configure_options(self):
        super(MRKNN, self).configure_options()
        self.add_file_option('--test_path', help='test file path')
        self.add_file_option('--output_path', help='path for storing output')
        self.add_passthrough_option('--k', type = 'int', default = 3, help=' value of k for knn')

    def mapper(self, _, sample):
        for line in open(self.options.test_path):

            test_features = line.split(';')
            dist = 0
            train_feat = sample.split(';')

            # use passegerid as key for reducer and remove it from feature list and remove '\n' from last column
            id = test_features[0]
            test_features = test_features[1:-1] + [test_features[-1][:-1]]
            dis_points = []
            y = []
            for i, test_feat_val in enumerate(test_features):

                # convert int & float into float
                try:
                    train_feat_val = float(train_feat[i])
                    test_feat_val = float(test_feat_val)
                except:
                    train_feat_val = str(train_feat[i])
                    test_feat_val = str(test_feat_val)

                # cal distance for each test sample
                if isinstance(test_feat_val, float):
                    dis_points.append([train_feat_val, test_feat_val])
                else:
                    # if string then dist =1 if same string
                    if train_feat_val == test_feat_val:
                        dist += 0
                    else:
                        dist += 1

            # calculate distance for numeric values
            obj = Distance()
            dist += obj.euclidean_dist(dis_points)

            yield (id, (dist, str(train_feat[-1])))

    def reducer(self, test_sample, res):

        res = list(res)
        res = sorted(res)
        labels = list(zip(*res)[1])
        top_k = labels[:self.options.k]
        dic = {}
        for val in top_k:
            if val not in dic:
                dic[val] = 0
            dic[val] += 1
        values = list(dic.values())
        keys = list(dic.keys())
        print test_sample, keys[values.index(max(values))]
        #f = open(train_path.replace(".csv", "") + '.txt', 'w')
        #f.write("\n".join(res))
        #yield (test_sample, keys[values.index(max(values))])


class Distance:

    def euclidean_dist(self, points):

        dist = 0
        for point in points:
            dist += (point[0] - point[1]) ** 2

        return dist

if __name__ == '__main__':
    MRKNN.run()