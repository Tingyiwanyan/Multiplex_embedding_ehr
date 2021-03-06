import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import groupby
#from evaluation import cal_auc



class NN_model():
    """
    Create shalow neural network model for EHR data
    """
    def __init__(self,kg,hetro_model,data_process):
        self.kg = kg
        self.data_process = data_process
        self.hetro_model = hetro_model
        self.train_data = self.data_process.train_patient
        self.test_data = self.data_process.test_patient
        self.length_train = len(self.train_data)
        self.length_train_hadm = len(data_process.train_hadm_id)
        self.batch_size = 16
        self.latent_dim = 100
        self.epoch = 6
        self.resolution = 0.0001
        self.threshold_diag = 0.06
        #self.item_size = len(list(kg.dic_item.keys()))
        #self.diagnosis_size = len(list(kg.dic_diag))
        #self.patient_size = len(list(kg.dic_patient))
        self.item_size = len(list(kg.dic_vital.keys()))
        self.demo_size = len(list(kg.dic_race.keys()))
        self.lab_size = len(list(kg.dic_lab.keys()))
        self.input_size = self.item_size+self.demo_size+self.lab_size
        self.input_seq = []
        self.positive_sample_size = 1
        self.negative_sample_size = self.batch_size
        """
        define shallow neural network
        """
        self.input_x = tf.placeholder(tf.float32, [None, self.input_size])
        self.input_x_positive = tf.placeholder(tf.float32,[None,self.positive_sample_size,self.input_size])
        self.input_x_negative = tf.placeholder(tf.float32,[None,self.negative_sample_size,self.input_size])

    def softmax_loss(self):
        """
        Implement softmax loss layer
        """
        self.hidden_rep = tf.layers.dense(inputs=self.input_x,
                                          units=self.latent_dim,
                                          kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                          activation=tf.nn.relu)
        self.batch_normed = tf.keras.layers.BatchNormalization()
        self.hidden_batch_normed = self.batch_normed(self.hidden_rep)
        self.output_layer = tf.layers.dense(inputs=self.hidden_rep,
                                           units=self.diagnosis_size,
                                           kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                           activation=tf.nn.elu)
        self.logit_softmax = tf.nn.softmax(self.output_layer)
        self.cross_entropy = tf.reduce_mean(tf.math.negative(
            tf.reduce_sum(tf.math.multiply(self.input_y_diag, tf.log(self.logit_softmax)), reduction_indices=[1])))

    def contrastive_loss(self):
        """
        Implement Contrastive Loss
        """
        """
        positive inner product
        """
        self.positive_broad = tf.broadcast_to(self.input_x,
                                              [self.batch_size,self.positive_sample_size,self.input_size])
        self.negative_broad = tf.broadcast_to(self.input_x,
                                              [self.batch_size,self.negative_sample_size,self.input_size])

        self.positive_broad_norm = tf.math.l2_normalize(self.positive_broad,axis=2)
        self.positive_sample_norm = tf.math.l2_normalize(self.input_x_positive,axis=2)

        self.positive_dot_prod = tf.multiply(self.positive_broad_norm,self.positive_sample_norm)
        self.positive_dot_prod_sum = tf.reduce_sum(tf.reduce_sum(self.positive_dot_prod, 2),1)

        """
        negative inner product
        """
        self.negative_broad_norm = tf.math.l2_normalize(self.negative_broad,axis=2)
        self.negative_sample_norm = tf.math.l2_normalize(self.input_x_negative,axis=2)

        self.negative_dot_prod = tf.multiply(self.negative_broad_norm,self.negative_sample_norm)
        self.negative_dot_prod_sum = tf.reduce_sum(tf.reduce_sum(self.negative_dot_prod,2),1)

        """
        Compute normalized probability and take log form
        """
        self.denominator_normalizer = tf.math.add(self.positive_dot_prod_sum,self.negative_dot_prod_sum)
        self.normalized_prob = tf.math.divide(self.positive_dot_prod_sum,self.denominator_normalizer)
        self.log_normalized_prob = tf.math.negative(tf.math.log(self.normalized_prob))



    def SGNN_loss_contrast(self):
        """
        implement sgnn loss contrast
        """
        negative_training_norm = tf.math.l2_normalize(self.x_negative_contrast, axis=2)

        skip_training = tf.broadcast_to(self.x_origin,
                                        [self.batch_size, self.negative_sample_size,
                                         self.latent_dim + self.latent_dim_demo])

        skip_training_norm = tf.math.l2_normalize(skip_training, axis=2)

        dot_prod = tf.multiply(skip_training_norm, negative_training_norm)

        dot_prod_sum = tf.reduce_sum(dot_prod, 2)

        sum_log_dot_prod = tf.math.log(tf.math.sigmoid(tf.math.negative(tf.reduce_mean(dot_prod_sum, 1))))

        positive_training = tf.broadcast_to(self.x_origin, [self.batch_size, self.positive_sample_size,
                                                            self.latent_dim + self.latent_dim_demo])

        positive_skip_norm = tf.math.l2_normalize(self.x_skip_contrast, axis=2)

        positive_training_norm = tf.math.l2_normalize(positive_training, axis=2)

        dot_prod_positive = tf.multiply(positive_skip_norm, positive_training_norm)

        dot_prod_sum_positive = tf.reduce_sum(dot_prod_positive, 2)

        sum_log_dot_prod_positive = tf.math.log(tf.math.sigmoid(tf.reduce_mean(dot_prod_sum_positive, 1)))

        self.negative_sum_contrast = tf.math.negative(
            tf.reduce_sum(tf.math.add(sum_log_dot_prod, sum_log_dot_prod_positive)))


    def get_batch_train(self, start_index):
        """
        get training batch data
        """
        train_one_batch_vital = np.zeros((data_length, self.item_size))
        train_one_batch_lab = np.zeros((data_length, self.lab_size))
        train_one_batch_demo = np.zeros((data_length, self.demo_size))
        self.real_logit = np.zeros((data_length, 1))


    def get_batch_train_origin(self, data_length, start_index, data):
        """
        get training batch data
        """
        train_one_batch_vital = np.zeros(
            (data_length, self.time_sequence, 1 + self.positive_lab_size + self.negative_lab_size, self.item_size))
        train_one_batch_lab = np.zeros(
            (data_length, self.time_sequence, 1 + self.positive_lab_size + self.negative_lab_size, self.lab_size))
        train_one_batch_icu_intubation = np.zeros((data_length,self.time_sequence,1+self.positive_lab_size+self.negative_lab_size,2))
        train_one_batch_demo = np.zeros(
            (data_length, 1 + self.positive_lab_size + self.negative_lab_size, self.demo_size))
        train_one_batch_com = np.zeros(
            (data_length, 1 + self.positive_lab_size + self.negative_lab_size, self.com_size))
        # train_one_batch_item = np.zeros((data_length,self.positive_lab_size+self.negative_lab_size,self.item_size))
        train_one_batch_mortality = np.zeros((data_length, 2, 2))
        one_batch_logit = np.zeros((data_length, 2))
        self.neg_patient_id = []
        for i in range(data_length):
            self.patient_id = data[start_index + i]
            self.neg_patient_id.append(self.patient_id)
        for i in range(data_length):
            self.patient_id = data[start_index + i]
            # if self.kg.dic_patient[self.patient_id]['item_id'].keys() == {}:
            #   index_increase += 1
            #  continue
            # index_batch += 1
            self.time_seq = self.kg.dic_patient[self.patient_id]['prior_time_vital'].keys()
            self.time_seq_int = [np.int(k) for k in self.time_seq]
            self.time_seq_int.sort()
            time_index = 0
            flag = self.kg.dic_patient[self.patient_id]['death_flag']
            """
            if flag == 0:
                one_batch_logit[i,0,0] = 1
                one_batch_logit[i,1,1] = 1
            else:
                one_batch_logit[i,0,1] = 1
                one_batch_logit[i,1,0] = 1
                self.real_logit[i] = 1
            """
            if flag == 0:
                train_one_batch_mortality[i, 0, :] = [1, 0]
                train_one_batch_mortality[i, 1, :] = [0, 1]
                one_batch_logit[i, 0] = 1
                self.real_logit[i,0] = 0
            else:
                train_one_batch_mortality[i, 0, :] = [0, 1]
                train_one_batch_mortality[i, 1, :] = [1, 0]
                one_batch_logit[i, 1] = 1
                self.real_logit[i,0] = 1

            self.get_positive_patient(self.patient_id)
            self.get_negative_patient(self.patient_id)
            train_one_data_vital = np.concatenate((self.patient_pos_sample_vital, self.patient_neg_sample_vital),
                                                  axis=1)
            train_one_data_lab = np.concatenate((self.patient_pos_sample_lab, self.patient_neg_sample_lab), axis=1)
            train_one_data_demo = np.concatenate((self.patient_pos_sample_demo, self.patient_neg_sample_demo), axis=0)
            train_one_data_com = np.concatenate((self.patient_pos_sample_com, self.patient_neg_sample_com), axis=0)
            train_one_data_icu_intubation = np.concatenate((self.patient_pos_sample_icu_intubation_label,self.patient_neg_sample_icu_intubation_label),axis=1)
            train_one_batch_vital[i, :, :, :] = train_one_data_vital
            train_one_batch_lab[i, :, :, :] = train_one_data_lab
            train_one_batch_demo[i, :, :] = train_one_data_demo
            train_one_batch_com[i, :, :] = train_one_data_com
            train_one_batch_icu_intubation[i,:,:,:] = train_one_data_icu_intubation



        return train_one_batch_vital, train_one_batch_lab, train_one_batch_demo, one_batch_logit, train_one_batch_mortality, train_one_batch_com,train_one_batch_icu_intubation

    def assign_value_vital(self, patientid):
        self.one_sample = np.zeros(self.item_size)
        self.times = []
        for j in self.kg.dic_patient[patientid]['prior_time_vital'].keys():
            for i in self.kg.dic_patient[patientid]['prior_time_vital'][str(j)].keys():
                mean = np.float(self.kg.dic_vital[i]['mean_value'])
                std = np.float(self.kg.dic_vital[i]['std'])
                ave_value = np.mean(
                    [np.float(k) for k in self.kg.dic_patient[patientid]['prior_time_vital'][str(j)][i]])
                index = self.kg.dic_vital[i]['index']
                #self.ave_item[index] = mean
                if std == 0:
                    self.one_sample[index] += 0
                    #self.freq_sample[index] += 1
                else:
                    if ave_value > mean+std:
                        self.one_sample[index] = 1
                    elif ave_value < mean-std:
                        self.one_sample[index] = -1
                    else:
                        self.one_sample[index] = (np.float(ave_value)-mean)/std

                    #self.one_sample[index] = np.float(ave_value) - mean / 3*std
                    self.freq_sample[index] += 1

        out_sample = self.one_sample / self.freq_sample
        for i in range(self.item_size):
            if math.isnan(out_sample[i]):
                out_sample[i] = 0

        return out_sample

    def assign_value_icu_intubation(self,patientid,start_time,end_time):
        one_sample_icu_intubation = np.zeros(2)
        if self.kg.dic_patient[patientid]['icu_label'] == 1:
            icu_hour = self.kg.dic_patient[patientid]['in_icu_hour']
            if icu_hour > start_time:
                one_sample_icu_intubation[0] = 1
        if self.kg.dic_patient[patientid]['intubation_label'] == 1:
            intubation_hour = self.kg.dic_patient[patientid]['intubation_hour']
            if intubation_hour > start_time and intubation_hour < end_time:
                one_sample_icu_intubation[1] = 1
            if intubation_hour < start_time:
                if self.kg.dic_patient[patientid]['extubation_label'] == 1:
                    extubation_hour = self.kg.dic_patient[patientid]['extubation_hour']
                    if extubation_hour > start_time:
                        one_sample_icu_intubation[1] = 1
                if self.kg.dic_patient[patientid]['extubation_label'] == 0:
                    one_sample_icu_intubation[1] = 1

        return one_sample_icu_intubation



    def assign_value_lab(self, patientid, start_time, end_time):
        self.one_sample_lab = np.zeros(self.lab_size)
        self.times_lab = []
        for j in self.kg.dic_patient[patientid]['prior_time_lab'].keys():
            for i in self.kg.dic_patient[patientid]['prior_time_lab'][str(j)].keys():
                if i[-1] == 'A':
                    continue
                if i == "EOSINO":
                    continue
                if i == "EOSINO_PERC":
                    continue
                if i == "BASOPHIL":
                    continue
                if i == "BASOPHIL_PERC":
                    continue
                mean = np.float(self.kg.dic_lab[i]['mean_value'])
                std = np.float(self.kg.dic_lab[i]['std'])
                #in_lier = np.where(np.array(self.kg.dic_patient[patientid]['prior_time_lab'][str(j)][i])<mean+3*std)[0]
                #in_lier_value = list(np.array(self.kg.dic_patient[patientid]['prior_time_lab'][str(j)][i]))#[in_lier])
                #if in_lier_value == []:
                    #ave_value = mean
                #else:
                ave_value = np.mean([np.float(k) for k in self.kg.dic_patient[patientid]['prior_time_lab'][str(j)][i]])
                    #ave_value = np.mean(
                        #[np.float(k) for k in in_lier_value])
                index = self.kg.dic_lab[i]['index']
                #self.ave_lab[index] = mean
                if std == 0:
                    self.one_sample_lab[index] += 0
                    #self.freq_sample_lab[index] += 1
                else:
                    if ave_value > mean+std:
                        self.one_sample[index] = 1
                    elif ave_value < mean-std:
                        self.one_sample[index] = -1
                    else:
                        self.one_sample[index] = (np.float(ave_value)-mean)/std
                    #self.one_sample_lab[index] += (np.float(ave_value) - mean) / std
                    self.freq_sample_lab[index] += 1

        out_sample_lab = self.one_sample_lab / self.freq_sample_lab
        for i in range(self.lab_size):
            if math.isnan(out_sample_lab[i]):
                out_sample_lab[i] = 0

        return out_sample_lab

    def assign_value_demo(self, patientid):
        one_sample = np.zeros(self.demo_size)
        for i in self.kg.dic_demographic[patientid]['race']:
            if i == 'race':
                race = self.kg.dic_demographic[patientid]['race']
                index = self.kg.dic_race[race]['index']
                one_sample[index] = 1
            elif i == 'Age':
                age = self.kg.dic_demographic[patientid]['Age']
                index = self.kg.dic_race['Age']['index']
                if age == 0:
                    one_sample[index] = age
                else:
                    one_sample[index] = (np.float(age) - self.kg.age_mean) / self.kg.age_std
            elif i == 'gender':
                gender = self.kg.dic_demographic[patientid]['gender']
                index = self.kg.dic_race[gender]['index']
                one_sample[index] = 1

        return one_sample

    def assign_value_com(self, patientid):
        one_sample = np.zeros(self.com_size)
        self.com_index = np.where(self.kg.com_mapping_ar[:, 0] == patientid)[0][0]
        deidentify_index = self.kg.com_mapping_ar[self.com_index][1]
        self.map_index = np.where(deidentify_index == self.kg.com_ar[:, 1])[0][0]
        one_sample[:] = [np.int(i) for i in self.kg.com_ar[self.map_index, 4:]]

        return one_sample

    def config_model(self):
        self.lstm_cell()
        self.demo_layer()
        # self.softmax_loss()
        self.build_dhgm_model()
        self.get_latent_rep_hetero()
        self.SGNN_loss()
        self.SGNN_loss_contrast()
        # self.train_step_neg = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.negative_sum)
        # self.train_step_neg = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(0.8*self.negative_sum+0.2*self.negative_sum_contrast)
        # self.train_step_cross_entropy = tf.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)
        self.train_step_neg = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.negative_sum_contrast)
        self.logit_sig = tf.compat.v1.layers.dense(inputs=self.x_origin_ce,
                                                   units=1,
                                                   kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                   activation=tf.nn.sigmoid)
        # self.logit_sig = tf.nn.softmax(self.output_layer)
        bce = tf.keras.losses.BinaryCrossentropy()
        self.cross_entropy = bce(self.logit_sig, self.input_y_logit)
        self.train_step_ce = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)
        self.train_step_combine_ce = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(
            self.cross_entropy + 0.2 * self.negative_sum_contrast)
        """
        focal loss
        """
        alpha = 0.25
        # self.focal_loss_ = -self.input_y_logit*tf.math.multiply((1-self.logit_sig)**self.gamma,tf.log(self.logit_sig))
        # self.focal_loss = tf.math.reduce_mean(self.focal_loss_)
        alpha_t = self.input_y_logit * alpha + (tf.ones_like(self.input_y_logit) - self.input_y_logit) * (1 - alpha)

        p_t = self.input_y_logit * self.logit_sig + (tf.ones_like(self.input_y_logit) - self.input_y_logit) * (
                    tf.ones_like(self.input_y_logit) - self.logit_sig) + tf.keras.backend.epsilon()

        self.focal_loss_ = - alpha_t * tf.math.pow((tf.ones_like(self.input_y_logit) - p_t), self.gamma) * tf.math.log(p_t)
        self.focal_loss = tf.reduce_mean(self.focal_loss_)
        self.train_step_fl = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.focal_loss)
        self.train_step_combine_fl = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(
            self.focal_loss + 0.2 * self.negative_sum_contrast)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()