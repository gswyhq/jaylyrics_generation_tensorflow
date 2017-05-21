# -*-  coding:utf-8 -*-
''' model for automatic speech recognition implemented in Tensorflow
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

from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
import pickle as cPickle

from preprocess import TextParser
from seq2seq_rnn import Model as Model_rnn
from utils import count_params
from utils import logging


class Trainer():
    def __init__(self):
        """
        训练函数需要完成的功能主要有提供用户可设置的超参数、读取配置文件、按照mini-batch进行批训练、
        使用saver保存模型参数、记录训练误差等等
        """
        # 使用argparse.ArgumentParser对象进行解析命令行参数或者设置默认参数
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', default='./data/',
                            help='设置包含 input.txt 文件的目录')

        parser.add_argument('--save_dir', default='./save/',
                            help='设置保存模型的路径')

        parser.add_argument('--log_dir', default='./log/',
                            help='设置保存输出日志的路径')

        parser.add_argument('--rnn_size', type=int, default=128,
                            help='设置RNN隐藏层神经元个数 ')

        parser.add_argument('--embedding_size', type=int, default=100,
                            help='设置字向量的大小')

        parser.add_argument('--num_layers', type=int, default=2,
                            help='设置RNN隐藏层数')

        parser.add_argument('--model', default='seq2seq_rnn',
                            help='设置模型')

        parser.add_argument('--rnncell', default='lstm',
                            help='RNN，GRU与LSTM语言模型选择')

        parser.add_argument('--attention', type=bool, default=False,
                            help='是否使用注意力模型')

        parser.add_argument('--batch_size', type=int, default=32,
                            help='设置批尺寸')

        parser.add_argument('--seq_length', type=int, default=16,
                            help='设置rnn系列长度')

        parser.add_argument('--num_epochs', type=int, default=10000,
                            help='设置epochs数（一个epoch是指把所有训练数据完整的过一遍）')

        parser.add_argument('--save_every', type=int, default=1000,
                            help='训练保存模型的频率')

        parser.add_argument('--grad_clip', type=float, default=20.,
                            help='设置反向传播权重梯度')

        parser.add_argument('--learning_rate', type=float, default=0.001,
                            help='设置学习率')

        parser.add_argument('--decay_rate', type=float, default=0.98,
                            help='设置衰减率')

        parser.add_argument('--keep', type=bool, default=False,
                            help='是否在原模型基础上继续训练')

        args = parser.parse_args()
        self.train(args)

    def train(self, args):
        ''' import data, train model, save model
	'''
        text_parser = TextParser(args.data_dir, args.batch_size, args.seq_length)
        args.vocab_size = text_parser.vocab_size

        # 加载已训练好的模型路径
        ckpt = tf.train.get_checkpoint_state(args.save_dir)

        # 我们需要提供是否继续训练的判断，也就说是从头开始训练还是导入一个已经训练过的模型继续训练
        if args.keep is True:
            # check if all necessary files exist 
            if os.path.exists(os.path.join(args.save_dir, 'config.pkl')) and \
                    os.path.exists(os.path.join(args.save_dir, 'words_vocab.pkl')) and \
                    ckpt and ckpt.model_checkpoint_path:
                with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
                    saved_model_args = cPickle.load(f)
                with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
                    saved_words, saved_vocab = cPickle.load(f)
            else:
                raise ValueError('configuration doesn"t exist!')

        if args.model == 'seq2seq_rnn':
            model = Model_rnn(args)
        else:
            # TO ADD OTHER MODEL
            raise ValueError("输入的seq2seu模型参数不对")

        trainable_num_params = count_params(model, mode='trainable')
        all_num_params = count_params(model, mode='all')
        args.num_trainable_params = trainable_num_params
        args.num_all_params = all_num_params
        print(args.num_trainable_params)
        print(args.num_all_params)
        with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
            cPickle.dump(args, f)
        with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
            cPickle.dump((text_parser.vocab_dict, text_parser.vocab_list), f)

        with tf.Session() as sess:
            if args.keep is True:
                print('Restoring')
                model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('Initializing')
                sess.run(model.initial_op)

            for e in range(args.num_epochs):
                start = time.time()
                # sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
                sess.run(tf.assign(model.lr, args.learning_rate))
                model.initial_state = tf.convert_to_tensor(model.initial_state)
                state = model.initial_state.eval()
                total_loss = []
                for b in range(text_parser.num_batches):

                    # 将X和Y数据feed到模型中去运行op并得到误差值
                    x, y = text_parser.next_batch()
                    print('flag')
                    feed = {model.input_data: x, model.targets: y, model.initial_state: state}
                    train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                    total_loss.append(train_loss)
                    print("{}/{} (epoch {}), train_loss = {:.3f}" \
                          .format(e * text_parser.num_batches + b, args.num_epochs * text_parser.num_batches, e, train_loss))
                    # 设置训练了多少个样本就保存一次参数、训练了多少个Epoch就保存一次
                    if (e * text_parser.num_batches + b) % args.save_every == 0 or (
                            e == args.num_epochs - 1 and b == text_parser.num_batches - 1):
                        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                        model.saver.save(sess, checkpoint_path, global_step=e)
                        print("model has been saved in:" + str(checkpoint_path))

                text_parser.reset_batch()


                end = time.time()
                delta_time = end - start
                ave_loss = np.array(total_loss).mean()
                logging(model, ave_loss, e, delta_time, mode='train')
                if ave_loss < 0.5:
                    break


if __name__ == '__main__':
    trainer = Trainer()
