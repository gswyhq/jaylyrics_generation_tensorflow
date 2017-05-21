# -*- coding:utf-8 -*-

''' Sequence generation implemented in Tensorflow
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
     
date:2016-12-07
'''


import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import numpy as np
import datetime
from utils import build_weight
from utils import random_pick

class Model():
    def __init__(self, args, infer=False):
        """
        数据预处理完成以后，接下来就是建立seq2seq模型了。建立模型主要分为三步：
        确定好编码器和解码器中cell的结构，即采用什么循环单元，多少个神经元以及多少个循环层；
        将输入数据转化成tensorflow的seq2seq.rnn_decoder需要的格式，并得到最终的输出以及最后一个隐含状态；
        将输出数据经过softmax层得到概率分布，并且得到误差函数，确定梯度下降优化器；

        由于tensorflow提供的rnncell共有三种，分别是RNN、GRU、LSTM，因此这里我们也提供三种选择，并且每一种都可以使用多层结构，
        即MultiRNNCell
        :param args: 
        :param infer: 
        """
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.rnncell == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.rnncell == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.rnncell == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("rnncell type not supported: {}".format(args.rnncell))

        cell = cell_fn(args.rnn_size)
        self.cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = self.cell.zero_state(args.batch_size, tf.float32)
        with tf.variable_scope('rnnlm'):
            softmax_w = build_weight([args.rnn_size, args.vocab_size],name='soft_w')
            softmax_b = build_weight([args.vocab_size],name='soft_b')
            word_embedding = build_weight([args.vocab_size, args.embedding_size],name='word_embedding')
            inputs_list = tf.split(1, args.seq_length, tf.nn.embedding_lookup(word_embedding, self.input_data))
            inputs_list = [tf.squeeze(input_, [1]) for input_ in inputs_list]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(word_embedding, prev_symbol)

        # 用于建立seq2seq的函数，rnn_decoder以及attention_decoder
        if not args.attention:
            outputs, last_state = seq2seq.rnn_decoder(inputs_list, self.initial_state, self.cell,
                                                      loop_function=loop if infer else None, scope='rnnlm')
            # rnn_decoder函数主要有四个参数
            # decoder_inputs其实就是输入的数据，要求的格式为一个list，并且list中的tensor大小应该为[batch_size，input_size]，
            # 换句话说这个list的长度就是seq_length；但我们原始的输入数据的维度为[args.batch_size, args.seq_length]，
            # 是不是感觉缺少了一个input_size维度，其实这个维度就是word_embedding的维度，或者说word2vec的大小，
            # 这里需要我们手动进行word_embedding，并且这个embedding矩阵是一个可以学习的参数

            # initial_state是cell的初始状态，其维度是[batch_size，cell.state_size]，
            # 由于rnn_cell模块提供了对状态的初始化函数，因此我们可以直接调用

            # cell就是我们要构建的解码器和编码器的cell，上面已经提过了。
            # 最后一个参数是loop_function，其作用是在生成的时候，我们需要把解码器上一时刻的输出作为下一时刻的输入，
            # 并且这个loop_function需要我们自己写

            # 其中outputs是与decoder_inputs同样维度的量，即每一时刻的输出；
            # last_state的维度是[batch_size，cell.state_size]，即最后时刻的所有cell的状态。
            # 接下来需要outputs来确定目标函数，而last-state的作用是作为抽样生成函数下一时刻的状态


        else:
            self.attn_length = 5
            self.attn_size = 32
            self.attention_states = build_weight([args.batch_size, self.attn_length, self.attn_size])
            outputs, last_state = seq2seq.attention_decoder(inputs_list, self.initial_state, self.attention_states, self.cell, loop_function=loop if infer else None, scope='rnnlm')

        self.final_state = last_state
        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)

        # tensorflow中提供了sequence_loss_by_example函数用于按照权重来计算整个序列中每个单词的交叉熵，
        # 返回的是每个序列的log-perplexity。为了使用sequence_loss_by_example函数，
        # 我们首先需要将outputs通过一个前向层，同时我们需要得到一个softmax概率分布

        # average loss for each word of each timestep
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.lr = tf.Variable(0.0, trainable=False)
        self.var_trainable_op = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, self.var_trainable_op),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)

        # train_op即为训练时需要运行的
        self.train_op = optimizer.apply_gradients(zip(grads, self.var_trainable_op))
        self.initial_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1)
        self.logfile = args.log_dir+str(datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')+'.txt').replace(' ','').replace('/','')
        self.var_op = tf.global_variables()

    def sample(self, sess, words, vocab, num=200, start=u'我们', sampling_type=1):

        state = sess.run(self.cell.zero_state(1, tf.float32))

        # 在抽样生成的时候，我们首先需要一个种子序列，同时在第一步的时候，我们需要向网络传入一个0的初始状态，
        # 并通过种子序列的第一个字得到下一个隐含状态，然后再结合种子的第二个字传入下一个隐含状态，直到种子序列传入完毕
        for word in start:
            x = np.zeros((1, 1))
            x[0, 0] = words[word]
        if not self.args.attention:
                feed = {self.input_data: x, self.initial_state:state}
                [probs, state] = sess.run([self.probs, self.final_state], feed)
        else:
            # TO BE UPDATED
            attention_states = sess.run(build_weight([self.args.batch_size,self.attn_length,self.attn_size],name='attention_states'))
            feed = {self.input_data: x, self.initial_state:state,self.attention_states:attention_states}
            [probs, state] = sess.run([self.probs, self.final_state], feed)

        # 其中start是种子序列，attention是判断是否加入了注意力机制。
        ret = start
        word = start[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = words[word]
            if not self.args.attention:
                    feed = {self.input_data: x, self.initial_state:state}
                    [probs, state] = sess.run([self.probs, self.final_state], feed)
            else:
                    feed = {self.input_data: x, self.initial_state:state,self.attention_states:attention_states}
                    [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            sample = random_pick(p,word,sampling_type)
            pred = vocab[sample]
            ret += pred
            word = pred
        return ret
