import tensorflow as tf
import numpy as np
import random
import math
import copy
from itertools import groupby
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

class linear_separate():
    """
    linear separate protocol
    """
    def __init__(self, out_patient_embedding, label):
        print("Im here in linear separate")
        self.data = out_patient_embedding
        self.data_length = self.data.shape[0]
        self.latent_dim = self.data.shape[1]
        self.label = label
        # self.train_num = np.int(np.floor(self.data_patient_num*self.train_percent))
        self.train_num = np.int(np.floor(self.data_length * 0.7))
        self.train_data = self.data[0:self.train_num,:]
        self.test_data = self.data[self.train_num:,:]
        self.train_logit = self.label[0:self.train_num,:]
        self.real_logit = self.label[self.train_num:,:]
        self.input_y_logit = tf.keras.backend.placeholder([None, 1])
        self.embedding = tf.keras.backend.placeholder([None,self.latent_dim])
        self.batch_size = 16
        self.epoch = 1

    def logistic_loss(self):
        self.logit_sig = tf.compat.v1.layers.dense(inputs=self.embedding,
                                                      units=1,
                                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                      activation=tf.nn.sigmoid)
        self.logistic_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_sig, labels=self.input_y_logit))
        self.train_step_log = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.logistic_loss)
        """
        bce = tf.keras.losses.BinaryCrossentropy()
        self.cross_entropy = bce(self.logit_sig, self.input_y_logit)
        self.train_step_ce = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)
        self.train_step_combine_ce = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.cross_entropy+0.2*self.log_normalized_prob)
        """
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    def train(self):
        iteration = np.int(np.floor(np.float(self.train_num) / self.batch_size))
        for j in range(self.epoch):
            print('epoch')
            print(j)
            # self.construct_knn_graph()
            for i in range(iteration):
                self.err_ = self.sess.run([self.logistic_loss, self.train_step_log],
                                          feed_dict={self.embedding:
                                                         self.train_data[i*self.batch_size:(i+1)*self.batch_size,:],
                                                     self.input_y_logit:self.train_logit[i*self.batch_size:(i+1)*self.batch_size,:]})
                print(self.err_[0])

    def test(self):
        self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.embedding:self.test_data})

        self.tp_correct = 0
        self.tp_neg = 0

        test_length = self.test_data.shape[0]
        for i in range(test_length):
            if self.real_logit[i, 0] == 1:
                self.tp_correct += 1
            if self.real_logit[i, 0] == 0:
                self.tp_neg += 1

        threshold = 0
        self.resolution = 0.01
        self.tp_total = []
        self.fp_total = []
        self.precision_total = []
        self.recall_total = []
        self.out_logit_integer = np.zeros(self.out_logit.shape[0])


        while (threshold < 1.01):
            tp_test = 0
            fp_test = 0
            fn_test = 0

            for i in range(test_length):
                if self.out_logit[i, 0] > threshold:
                    self.out_logit_integer[i] = 1

            for i in range(test_length):
                if self.real_logit[i, 0] == 1 and self.out_logit[i, 0] > threshold:
                    tp_test += 1
                if self.real_logit[i, 0] == 0 and self.out_logit[i, 0] > threshold:
                    fp_test += 1
                if self.out_logit[i, 0] < threshold and self.real_logit[i, 0] == 1:
                    fn_test += 1

            tp_rate = tp_test / self.tp_correct
            fp_rate = fp_test / self.tp_neg

            if (tp_test + fp_test) == 0:
                precision_test = 1.0
            else:
                precision_test = np.float(tp_test) / (tp_test + fp_test)
            recall_test = np.float(tp_test) / (tp_test + fn_test)

            # precision_test = precision_score(np.squeeze(self.real_logit), self.out_logit_integer, average='macro')
            # recall_test = recall_score(np.squeeze(self.real_logit), self.out_logit_integer, average='macro')
            self.tp_total.append(tp_rate)
            self.fp_total.append(fp_rate)
            self.precision_total.append(precision_test)
            self.recall_total.append(recall_test)
            threshold += self.resolution
            self.out_logit_integer = np.zeros(self.out_logit.shape[0])

    def cal_auc(self):
        self.area = 0
        self.tp_total.sort()
        self.fp_total.sort()
        for i in range(len(self.tp_total) - 1):
            x = self.fp_total[i + 1] - self.fp_total[i]
            y = (self.tp_total[i + 1] + self.tp_total[i]) / 2
            self.area += x * y

    def cal_auprc(self):
        self.area_auprc = 0
        #self.precision_total.sort()
        #self.recall_total.sort()
        for i in range(len(self.precision_total)-1):
            x = self.recall_total[i + 1] - self.recall_total[i]
            y = (self.precision_total[i + 1] + self.precision_total[i]) / 2
            self.area_auprc += x * y




