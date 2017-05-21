# -*- coding:utf-8 -*-
''' lyrics generation
author:

      iiiiiiiiiiii            iiiiiiiiiiii         !!!!!!!             !!!!!!    
      #        ###            #        ###           ###        I#        #:     
      #      ###              #      I##;             ##;       ##       ##      
            ###                     ###               !##      ####      #       
           ###                     ###                 ###    ## ###    #'       
         !##;                    `##%                   ##;  ##   ###  ##        
        ###                     ###                     $## `#     ##  #         
       ###        #            ###        #              ####      ####;         
     `###        -#           ###        `#               ###       ###          
     ##############          ##############               `#         #     
     
date:2016-12-01
'''
import os
import numpy as np
import re
import collections
import pickle as cPickle

def preprocess(c):
    """
    数据预处理，删除所有的非中文字符，并且使用空格分割相邻的句子
    :param c:
    :return:
    """
    reg = re.compile(r"[\s+]")
    c = reg.sub(' ', c)
    reg = re.compile(r"[^\u4e00-\u9fa5\s]")
    c = reg.sub('', c)
    c = c.strip()
    return c

class TextParser():
    def __init__(self, data_dir='./data/', batch_size=8, seq_length=10):
        '''
        初始化基本目录，batch_size和序列长度
        :param data_dir: 数据文件路径
        :param batch_size:
        :param seq_length:
        '''
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.input_file = os.path.join(data_dir, "lyrics.txt")  # 已处理过的歌词文件
        self.vocab_file = os.path.join(data_dir, "vocab.pkl")
        self.context_file = os.path.join(data_dir, "context.npy")

        if not (os.path.exists(self.vocab_file) and os.path.exists(self.context_file)):
            print("building dataset...")
            self.build_dataset()
        else:
            print("loading dataset...")
            self.load_dataset()
        self.init_batches()

    def build_dataset(self):
        ''' parse all sentences to build a vocabulary
        dictionary and vocabulary list
    '''
        with open(self.input_file, "r",encoding='utf-8') as f:
            data = f.read()
        # In[21]: ts = collections.Counter(['今天', '天气', '今天', '今天', '天气', '今天', '不错', '今天', '天气', '今天', '今天', '不错', '呢我'])
        # In[22]: ts
        # Out[22]: Counter({'不错': 2, '今天': 7, '呢我': 1, '天气': 3})
        # In[23]: ts.most_common(3)
        # Out[23]: [('今天', 7), ('天气', 3), ('不错', 2)]
        # 获取词频，取词频前vocabulary_size-1个词
        wordCounts = collections.Counter(data)
        self.vocab_list = [x[0] for x in wordCounts.most_common()]
        self.vocab_size = len(self.vocab_list)
        self.vocab_dict = {x: i for i, x in enumerate(self.vocab_list)}  # 字及其id
        with open(self.vocab_file, 'wb',encoding='utf-8') as f:
            cPickle.dump(self.vocab_list, f)
        self.context = np.array(list(map(self.vocab_dict.get, data)))  # 字对应id组成的列表
        print (self.context)
        np.save(self.context_file, self.context)


    def load_dataset(self):
        ''' if vocabulary has existed, we just load it
    '''
        with open(self.vocab_file, 'rb') as f:
            self.vocab_list = cPickle.load(f)
        self.vocab_size = len(self.vocab_list)
        self.vocab_dict = dict(zip(self.vocab_list, range(len(self.vocab_list))))
        self.context = np.load(self.context_file)
        self.num_batches = int(self.context.size / (self.batch_size * self.seq_length))

    def init_batches(self):
        ''' 然后确定我们要建立的每对样本的长度以及训练时候的batch_size大小，
        进而把数据集分成很多个mini-batch，可以在训练的时候依次读取。
        这里需要注意的是，为了预处理方便，我们选择了固定长度作为样本序列的长度，并且让X和Y的长度一致，
        从数据集中选取X和Y的时候每次滑动步长为1，间隔也为1

        可以看到Y的最后一个数是设置为X的第一个数，因此我们在数据集的开头插入了一个空格使得整体连贯。
        pointer是作为标记来用的，它的作用是标记当前训练的是哪一个mini-batch，如果所有mini-batch都训练过了，
        即完成了一个Epoch，那么pointer将置零
        '''
        self.num_batches = int(self.context.size / (self.batch_size * self.seq_length))
        self.context = self.context[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.context
        ydata = np.copy(self.context)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.pointer = 0

    def reset_batch(self):
        self.pointer = 0

    def next_batch(self):
        '''pointer是作为标记来用的，它的作用是标记当前训练的是哪一个mini-batch，如果所有mini-batch都训练过了，
        即完成了一个Epoch，那么pointer将置零
        '''
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

# test code
if __name__ == '__main__':
    t = TextParser(data_dir='/home/gswyhq/data/Neural_Writing_Machine')
    # t.build_dataset()
