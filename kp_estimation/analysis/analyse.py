# coding:utf8

import pandas as pd
from kp_estimation.train.data_processor import DataAggregator
from kp_estimation.train.model import FastTextModel
import os
import sys
sys.path.append('../')


class DataAnalyst:
    def __init__(self, subject, grade):
        self.subject = subject
        self.grade = grade

    def indicator(self, subject, grade, nums, model_path, data_path):
        ftm = FastTextModel(None)
        ftm.load_model(model_path, label_prefix='__label__')

        data_name = os.path.join(data_path, '%s_%s_%s.txt' % (subject, grade, nums))
        df = pd.read_csv(data_name,
                         header=None,
                         delimiter="\t",
                         names=["content", "num", 'kp_ids'],
                         dtype='object'
                         )

        result = ftm.predict(df['content'].values, recommend_kp_num=4)

        label = df['kp_ids'].tolist()

        if len(label) == 0:
            # 本科目年级没有该知识点数的题
            return 0, 0

        count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for i in range(len(label)):
            tmp = 0

            now_label = label[i].split(',')
            now_result = result[i]
            # print(now_label, now_result)

            for j in now_result:
                if now_label.count(j):
                    tmp += 1
            count[tmp] += 1

        print('所含题数\t%s' % len(label))

        cov = 0

        acc = count[nums]
        for k, v in count.items():
            if 0 < k <= nums:
                cov += count[k]

        print(count, '子精确数,', acc, '子覆盖数', cov, '子精确率,', float(acc) / len(label), \
              '子覆盖率', float(cov) / len(label))
        print()

        return acc, cov

    def analyse(self, inpath, outpath, model_path, subject, grade, nums=4):
        da = DataAggregator(inpath, outpath)
        da.run(["content", "kp_id"], nums=nums, subject=subject, grade=grade)
        print('data agg done')
        length = da.data_len
        #
        if length == 0:
            print('无此科目年级题')
            return

        acc = 0
        cov = 0
        for i in range(1, nums + 1):
            print('题目含有知识点数\t%s' % i)

            a, c = self.indicator(subject, grade, i, model_path, outpath)
            acc += a
            cov += c

        print('总精确数', acc, '总覆盖数', cov)
        # print('总精确率', float(acc) / length, '总覆盖率', float(cov) / length)
        print('-' * 60)


if __name__ == '__main__':
    da = DataAnalyst('020', '7')
    # da.analyse('/opt/kp_estimation/kpest_train/data/020_7_fasttext_test.txt',
    #            './',
    #            '/opt/kp_estimation/kpest_train/model_1206/020_7_fasttext.model.bin',
    #            '020', '7', 5)

    da.analyse('../../dataset/020_7_shuffle_test.csv',
               './',
               '../model/020_7_fasttext.model.bin',
               '020',
               '7', 4)
