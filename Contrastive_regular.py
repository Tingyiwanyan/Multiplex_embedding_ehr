import tensorflow as tf
import numpy as np
import random
import math
import copy
from itertools import groupby
import pandas as pd


class con_regular():
    """
    Create dynamic HGM model
    """

    def __init__(self, kg, data_process):
        print("Im here in death")
        # tf.compat.v1.disable_v2_behavior()
        # tf.compat.v1.disable_eager_execution()
        self.kg = kg
        self.data_process = data_process
        # self.hetro_model = hetro_model
        self.train_data_whole = self.data_process.train_patient_whole
        self.test_data_whole = self.data_process.test_patient_whole
        self.train_data = self.train_data_whole[0]
        self.test_data = self.test_data_whole[0]
        self.softmax_weight_threshold = 0.1
        #self.length_train = len(self.train_data)
        #self.length_test = len(self.test_data)
        self.batch_size = 16
        self.time_sequence = 4
        self.time_step_length = 6
        self.predict_window_prior = self.time_sequence * self.time_step_length
        self.latent_dim_cell_state = 100
        self.latent_dim_att = 100
        self.latent_dim_demo = 50
        self.epoch = 2
        self.item_size = len(list(kg.dic_vital.keys()))
        self.demo_size = len(list(kg.dic_race.keys()))
        self.lab_size = len(list(kg.dic_lab.keys()))
        self.latent_dim = self.item_size + self.lab_size
        self.com_size = 12
        self.input_seq = []
        self.threshold = 0.5
        self.positive_lab_size = 2
        self.negative_lab_size = 1
        self.positive_sample_size = self.positive_lab_size# + 1
        # self.positive_sample_size = 2
        self.negative_sample_size = self.negative_lab_size# + 1
        # self.negative_sample_size = 2
        self.neighbor_pick_skip = 5
        self.neighbor_pick_neg = 10
        self.neighbor_death = len(kg.dic_death[1])
        self.neighbor_discharge = len(kg.dic_death[0])
        """
        define LSTM variables
        """
        self.init_hiddenstate = tf.keras.backend.placeholder(
            [None, 1 + self.positive_lab_size + self.negative_lab_size, self.latent_dim])
        self.input_y_logit = tf.keras.backend.placeholder([None, 2])
        self.input_x_vital = tf.keras.backend.placeholder(
            [None, self.time_sequence, 1 + self.positive_lab_size + self.negative_lab_size, self.item_size])
        self.input_x_lab = tf.keras.backend.placeholder(
            [None, self.time_sequence, 1 + self.positive_lab_size + self.negative_lab_size, self.lab_size])
        self.input_icu_intubation = tf.keras.backend.placeholder(
            [None,self.time_sequence,1+self.positive_lab_size+self.negative_lab_size,2])
        self.input_x = tf.concat([self.input_x_vital, self.input_x_lab], 3)
        #self.input_x = tf.concat([self.input_x,self.input_icu_intubation],3)
        self.input_x_demo = tf.keras.backend.placeholder(
            [None, 1 + self.positive_lab_size + self.negative_lab_size, self.demo_size])
        self.input_x_com = tf.keras.backend.placeholder(
            [None, 1 + self.positive_lab_size + self.negative_lab_size, self.com_size])
        # self.input_x_demo = tf.concat([self.input_x_demo_,self.input_x_com],2)
        self.init_forget_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_info_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_cell_state = tf.keras.initializers.he_normal(seed=None)
        self.init_output_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_forget_gate_weight = tf.keras.initializers.he_normal(seed=None)
        self.init_info_gate_weight = tf.keras.initializers.he_normal(seed=None)
        self.init_cell_state_weight = tf.keras.initializers.he_normal(seed=None)
        self.weight_forget_gate = \
            tf.Variable(
                self.init_forget_gate(shape=(self.item_size + self.lab_size + self.latent_dim, self.latent_dim)))
        self.weight_info_gate = \
            tf.Variable(self.init_info_gate(shape=(self.item_size + self.lab_size + self.latent_dim, self.latent_dim)))
        self.weight_cell_state = \
            tf.Variable(self.init_cell_state(shape=(self.item_size + self.lab_size + self.latent_dim, self.latent_dim)))
        self.weight_output_gate = \
            tf.Variable(
                self.init_output_gate(shape=(self.item_size + self.lab_size + self.latent_dim, self.latent_dim)))
        self.bias_forget_gate = tf.Variable(self.init_forget_gate_weight(shape=(self.latent_dim,)))
        self.bias_info_gate = tf.Variable(self.init_info_gate_weight(shape=(self.latent_dim,)))
        self.bias_cell_state = tf.Variable(self.init_cell_state_weight(shape=(self.latent_dim,)))
        self.bias_output_gate = tf.Variable(self.init_output_gate(shape=(self.latent_dim,)))

        """
        Define LSTM variables plus attention
        """
        self.init_hiddenstate_att = tf.keras.backend.placeholder([None,
                                                                  1 + self.positive_lab_size + self.negative_lab_size + self.neighbor_pick_skip + self.neighbor_pick_neg,
                                                                  self.latent_dim])
        self.input_x_vital_att = tf.keras.backend.placeholder([None, self.time_sequence,
                                                               1 + self.positive_lab_size + self.negative_lab_size + self.neighbor_pick_skip + self.neighbor_pick_neg,
                                                               self.item_size])
        self.input_x_lab_att = tf.keras.backend.placeholder([None, self.time_sequence,
                                                             1 + self.positive_lab_size + self.negative_lab_size + self.neighbor_pick_skip + self.neighbor_pick_neg,
                                                             self.lab_size])
        self.input_x_att = tf.concat([self.input_x_vital_att, self.input_x_lab_att], 3)
        self.input_x_demo_att = tf.keras.backend.placeholder([None,
                                                              1 + self.positive_lab_size + self.negative_lab_size + self.neighbor_pick_skip + self.neighbor_pick_neg,
                                                              self.demo_size])

        """
        Define relation model
        """
        self.shape_relation = (self.latent_dim + self.latent_dim_demo,)
        self.init_mortality = tf.keras.initializers.he_normal(seed=None)
        self.init_lab = tf.keras.initializers.he_normal(seed=None)
        self.shape_relation_patient = (self.item_size+self.lab_size,self.latent_dim+self.latent_dim_demo)
        """
        Define parameters
        """
        self.mortality = tf.keras.backend.placeholder([None, 2, 2])
        self.Death_input = tf.keras.backend.placeholder([1, 2])
        self.init_weight_mortality = tf.keras.initializers.he_normal(seed=None)
        self.weight_mortality = \
            tf.Variable(self.init_weight_mortality(shape=(2, self.latent_dim + self.latent_dim_demo)))
        self.bias_mortality = tf.Variable(self.init_weight_mortality(shape=(self.item_size+self.lab_size + self.latent_dim_demo,)))

        self.lab_test = \
            tf.keras.backend.placeholder([None, self.positive_lab_size + self.negative_lab_size, self.item_size])
        self.weight_lab = \
            tf.Variable(self.init_weight_mortality(shape=(self.item_size, self.latent_dim)))
        self.bias_lab = tf.Variable(self.init_weight_mortality(shape=(self.latent_dim,)))
        """
        relation type 
        """
        self.relation_mortality = tf.Variable(self.init_mortality(shape=self.shape_relation))
        self.relation_lab = tf.Variable(self.init_lab(shape=self.shape_relation))

        """
        Multi relation type for patients
        """
        #self.relation_patients = tf.Variable(self.init_mortality(shape=self.shape_relation_patient))
        self.relation_patients = tf.keras.backend.placeholder([self.item_size+self.lab_size,self.latent_dim+self.latent_dim_demo])
        """
        Define orthogonal relation space
        """
        #self.relation_patients = self.orthog_init(shape=self.shape_relation_patient)
        self.relation_weight = tf.keras.backend.placeholder([None,self.positive_lab_size + self.negative_lab_size, self.item_size+self.lab_size])
        self.relation_weight_softmax = tf.nn.softmax(self.relation_weight)


        """
        Define attention mechanism
        """
        self.init_weight_att_W = tf.keras.initializers.he_normal(seed=None)
        self.init_weight_vec_a = tf.keras.initializers.he_normal(seed=None)
        self.weight_att_W = tf.Variable(self.init_weight_att_W(
            shape=(self.latent_dim + self.latent_dim_demo, self.latent_dim_att + self.latent_dim_demo)))
        self.weight_vec_a = tf.Variable(
            self.init_weight_vec_a(shape=(2 * (self.latent_dim_att + self.latent_dim_demo), 1)))

        """
        Define attention on sample neighbors
        """
        self.init_weight_vec_a_neighbor = tf.keras.initializers.he_normal(seed=None)
        self.weight_vec_a_neighbor = tf.Variable(
            self.init_weight_vec_a_neighbor(shape=(self.latent_dim + self.latent_dim_demo, 1)))

        """
        Define attention on Retain model for time
        """
        self.init_retain_b = tf.keras.initializers.he_normal(seed=None)
        self.init_retain_weight = tf.keras.initializers.he_normal(seed=None)
        self.weight_retain_w = tf.Variable(self.init_retain_weight(shape=(self.latent_dim, 1)))

        """
        Define attention on Retain model for feature variable
        """
        self.init_retain_variable_b = tf.keras.initializers.he_normal(seed=None)
        self.bias_retain_variable_b = tf.Variable(self.init_retain_variable_b(shape=(self.latent_dim,)))
        self.init_retain_variable_w = tf.keras.initializers.he_normal(seed=None)
        self.weight_retain_variable_w = tf.Variable(
            self.init_retain_variable_w(shape=(self.latent_dim, self.latent_dim)))

        """
        Define input projection
        """
        self.init_projection_b = tf.keras.initializers.he_normal(seed=None)
        self.bias_projection_b = tf.Variable(self.init_projection_b(shape=(self.latent_dim,)))
        self.init_projection_w = tf.keras.initializers.he_normal(seed=None)
        self.weight_projection_w = tf.Variable(
            self.init_projection_w(shape=(self.lab_size+self.item_size, self.latent_dim)))

    def lstm_cell(self):
        cell_state = []
        hidden_rep = []
        self.project_input = tf.math.add(tf.matmul(self.input_x, self.weight_projection_w), self.bias_projection_b)
        #self.project_input = tf.matmul(self.input_x, self.weight_projection_w)
        for i in range(self.time_sequence):
            x_input_cur = tf.gather(self.project_input, i, axis=1)
            if i == 0:
                concat_cur = tf.concat([self.init_hiddenstate, x_input_cur], 2)
            else:
                concat_cur = tf.concat([hidden_rep[i - 1], x_input_cur], 2)
            forget_cur = \
                tf.math.sigmoid(tf.math.add(tf.matmul(concat_cur, self.weight_forget_gate), self.bias_forget_gate))
            info_cur = \
                tf.math.sigmoid(tf.math.add(tf.matmul(concat_cur, self.weight_info_gate), self.bias_info_gate))
            cellstate_cur = \
                tf.math.tanh(tf.math.add(tf.matmul(concat_cur, self.weight_cell_state), self.bias_cell_state))
            info_cell_state = tf.multiply(info_cur, cellstate_cur)
            if not i == 0:
                forget_cell_state = tf.multiply(forget_cur, cell_state[i - 1])
                cellstate_cur = tf.math.add(forget_cell_state, info_cell_state)
            output_gate = \
                tf.nn.relu(tf.math.add(tf.matmul(concat_cur, self.weight_output_gate), self.bias_output_gate))
            hidden_current = tf.multiply(output_gate, cellstate_cur)
            cell_state.append(cellstate_cur)
            hidden_rep.append(hidden_current)

        self.hidden_last = hidden_rep[self.time_sequence - 1]
        for i in range(self.time_sequence):
            hidden_rep[i] = tf.expand_dims(hidden_rep[i], 1)
        self.hidden_rep = tf.concat(hidden_rep, 1)
        self.check = concat_cur

    def demo_layer(self):
        self.Dense_demo = tf.compat.v1.layers.dense(inputs=self.input_x_demo,
                                                    units=self.latent_dim_demo,
                                                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                    activation=tf.nn.relu)

    def demo_layer_att(self):
        self.Dense_demo = tf.compat.v1.layers.dense(inputs=self.input_x_demo_att,
                                                    units=self.latent_dim_demo,
                                                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                    activation=tf.nn.relu)

    def softmax_loss(self):
        """
        Implement softmax loss layer
        """
        idx_origin = tf.constant([0])
        self.hidden_last_comb = tf.concat([self.hidden_last, self.Dense_demo], 2)
        self.patient_lstm = tf.gather(self.hidden_last_comb, idx_origin, axis=1)
        self.output_layer = tf.compat.v1.layers.dense(inputs=self.patient_lstm,
                                                      units=2,
                                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                      activation=tf.nn.relu)
        # self.logit_sig = tf.math.sigmoid(self.output_layer)
        self.logit_sig = tf.nn.softmax(self.output_layer)
        bce = tf.keras.losses.BinaryCrossentropy()
        self.cross_entropy = bce(self.logit_sig, self.input_y_logit)
        # self.L2_norm = tf.math.square(tf.math.subtract(self.input_y_logit,self.logit_sig))
        # self.cross_entropy = tf.reduce_mean(tf.reduce_sum(self.L2_norm,axis=1),axis=0)

    def build_dhgm_model(self):
        """
        Build dynamic HGM model
        """
        #self.Dense_patient = tf.expand_dims(self.hidden_last,1)
        #self.Dense_patient = tf.concat([self.hidden_last,self.Dense_demo],2)


        self.hidden_att_e = tf.matmul(self.hidden_rep,self.weight_retain_w)
        self.hidden_att_e_softmax = tf.nn.softmax(self.hidden_att_e,1)
        self.hidden_att_e_broad = tf.broadcast_to(self.hidden_att_e_softmax,[tf.shape(self.input_x_vital)[0],
                                                                             self.time_sequence,1+self.positive_lab_size+self.negative_lab_size,self.latent_dim])


        self.hidden_att_e_variable = tf.math.sigmoid(
            tf.math.add(tf.matmul(self.hidden_rep, self.weight_retain_variable_w), self.bias_retain_variable_b))
        # self.hidden_att_e_softmax = tf.nn.softmax(self.hidden_att_e, -1)
        self.parameter_mul = tf.multiply(self.hidden_att_e_broad,self.hidden_att_e_variable)
        self.hidden_mul_variable = tf.multiply(self.parameter_mul, self.project_input)
        # self.hidden_final = tf.reduce_sum(self.hidden_mul, 1)
        self.hidden_final = tf.reduce_sum(self.hidden_mul_variable, 1)
        self.Dense_patient = tf.concat([self.hidden_final, self.Dense_demo], 2)
        #self.Dense_patient = tf.concat([self.hidden_mul_variable, self.Dense_demo], 2)



        #self.Dense_patient = self.hidden_last_comb
        # self.Dense_patient = tf.expand_dims(self.hidden_rep,2)


        self.Dense_mortality_ = \
            tf.nn.relu(tf.math.add(tf.matmul(self.mortality, self.weight_mortality), self.bias_mortality))

        self.Dense_mortality = tf.math.subtract(self.Dense_mortality_, self.relation_mortality)

        self.Dense_death_rep_ = \
            tf.nn.relu(tf.math.add(tf.matmul(self.Death_input, self.weight_mortality), self.bias_mortality))

        self.Dense_death_rep = tf.math.subtract(self.Dense_death_rep_, self.relation_mortality)

    def get_latent_rep_hetero(self):
        """
        Prepare data for SGNS loss function
        """
        idx_origin = tf.constant([0])
        self.x_origin = tf.gather(self.Dense_patient, idx_origin, axis=1)
        # self.x_origin = self.hidden_last

        idx_skip_mortality = tf.constant([0])
        self.x_skip_mor = tf.gather(self.Dense_mortality, idx_skip_mortality, axis=1)
        idx_neg_mortality = tf.constant([1])
        self.x_negative_mor = tf.gather(self.Dense_mortality, idx_neg_mortality, axis=1)

        """
        item_idx_skip = tf.constant([i+1 for i in range(self.positive_lab_size)])
        self.x_skip_item = tf.gather(self.Dense_lab,item_idx_skip,axis=1)
        item_idx_negative = tf.constant([i+self.positive_lab_size+1 for i in range(self.negative_lab_size)])
        self.x_negative_item = tf.gather(self.Dense_lab,item_idx_negative,axis=1)
        self.x_skip = tf.concat([self.x_skip,self.x_skip_item],axis=1)
        self.x_negative = tf.concat([self.x_negative,self.x_negative_item],axis=1)
        """
        patient_idx_skip = tf.constant([i + 1 for i in range(self.positive_lab_size)])
        self.x_skip_patient = tf.gather(self.Dense_patient, patient_idx_skip, axis=1)
        patient_idx_negative = tf.constant([i + self.positive_lab_size + 1 for i in range(self.negative_lab_size)])
        self.x_negative_patient = tf.gather(self.Dense_patient, patient_idx_negative, axis=1)



        # self.process_patient_att()

        #self.x_skip = tf.concat([self.x_skip_mor, self.x_skip_patient], axis=1)
        #self.x_negative = tf.concat([self.x_negative_mor, self.x_negative_patient], axis=1)
        self.x_skip = self.x_skip_mor
        self.x_negative = self.x_negative_mor
        self.x_skip_contrast_ = self.x_skip_patient
        self.x_negative_contrast_ = self.x_negative_patient

        self.x_skip_contrast_ = tf.expand_dims(self.x_skip_contrast_,axis=2)
        self.x_negative_contrast_ = tf.expand_dims(self.x_negative_contrast_,axis=2)

        self.x_skip_contrast = tf.broadcast_to(self.x_skip_contrast_, [self.batch_size, self.positive_sample_size ,self.item_size + self.lab_size,
                                        self.latent_dim + self.latent_dim_demo])

        self.x_negative_contrast = tf.broadcast_to(self.x_negative_contrast_,
                        [self.batch_size, self.negative_sample_size,self.item_size + self.lab_size,
                         self.latent_dim + self.latent_dim_demo])

        self.relation_patients = tf.expand_dims(self.relation_patients,axis=0)
        self.relation_patients = tf.expand_dims(self.relation_patients,axis=0)
        self.relation_patients_broad_pos = tf.broadcast_to(self.relation_patients, [self.batch_size, self.positive_sample_size ,self.item_size + self.lab_size,
                                        self.latent_dim + self.latent_dim_demo])



        self.relation_patients_broad_neg = tf.broadcast_to(self.relation_patients,[self.batch_size, self.negative_sample_size,self.item_size + self.lab_size,
                         self.latent_dim + self.latent_dim_demo])

        self.relation_patients_broad_origin = tf.broadcast_to(self.relation_patients,[self.batch_size,1,self.item_size+self.lab_size,
                                                                                      self.latent_dim+self.latent_dim_demo])

        """
        extract softmax weights
        """
        softmax_weight_idx_skip = tf.constant([i for i in range(self.positive_lab_size)])
        self.softmax_skip_patient = tf.gather(self.relation_weight_softmax, softmax_weight_idx_skip, axis=1)
        softmax_weight_idx_negative = tf.constant([i + self.positive_lab_size for i in range(self.negative_lab_size)])
        self.softmax_negative_patient = tf.gather(self.relation_weight_softmax, softmax_weight_idx_negative, axis=1)

        self.softmax_skip_patient = tf.expand_dims(self.softmax_skip_patient,axis=3)
        self.softmax_negative_patient = tf.expand_dims(self.softmax_negative_patient,axis=3)

        self.softmax_weight_pos_broad = tf.broadcast_to(self.softmax_skip_patient, [self.batch_size, self.positive_sample_size ,self.item_size + self.lab_size,
                                        self.latent_dim + self.latent_dim_demo])

        self.softmax_weight_neg_broad = tf.broadcast_to(self.softmax_negative_patient,
                                                           [self.batch_size, self.negative_sample_size,
                                                            self.item_size + self.lab_size,
                                                            self.latent_dim + self.latent_dim_demo])

        self.relation_patients_pos_add = tf.math.add(self.x_skip_contrast,self.relation_patients_broad_pos)
        self.relation_patients_neg_add = tf.math.add(self.x_negative_contrast,self.relation_patients_broad_neg)
        #self.relation_patients_neg_add = self.x_negative_contrast

        self.relation_patients_pos_add_weight = tf.math.multiply(self.relation_patients_pos_add,self.softmax_weight_pos_broad)
        self.relation_patients_neg_add_weight = tf.math.multiply(self.relation_patients_neg_add,self.softmax_weight_neg_broad)
        #self.relation_patients_neg_add_weight = self.x_negative_contrast

        """
        project x_origin into relation translation space
        """
        self.x_origin_ = tf.expand_dims(self.x_origin,axis=2)
        self.x_origin_contrast_ = tf.broadcast_to(self.x_origin_, [self.batch_size, 1, self.item_size + self.lab_size,
                                        self.latent_dim + self.latent_dim_demo])

        self.x_origin_contrast = tf.math.add(self.x_origin_contrast_,self.relation_patients_broad_origin)

    def get_positive_patient(self, center_node_index):
        self.patient_pos_sample_vital = np.zeros((self.time_sequence, self.positive_lab_size + 1, self.item_size))
        self.patient_pos_sample_lab = np.zeros((self.time_sequence, self.positive_lab_size + 1, self.lab_size))
        self.patient_pos_sample_icu_intubation_label = np.zeros((self.time_sequence, self.positive_lab_size+1, 2))
        self.patient_pos_sample_demo = np.zeros((self.positive_lab_size + 1, self.demo_size))
        self.patient_pos_sample_com = np.zeros((self.positive_lab_size + 1, self.com_size))
        if self.kg.dic_patient[center_node_index]['death_flag'] == 0:
            flag = 0
            neighbor_patient = self.kg.dic_death[0]
        else:
            flag = 1
            neighbor_patient = self.kg.dic_death[1]
        time_seq = self.kg.dic_patient[center_node_index]['prior_time_vital'].keys()
        time_seq_int = [np.int(k) for k in time_seq]
        time_seq_int.sort()
        # time_index = 0
        # for j in self.time_seq_int:
        for j in range(self.time_sequence):
            # if time_index == self.time_sequence:
            #    break
            if flag == 0:
                pick_death_hour = self.kg.dic_patient[center_node_index]['pick_time']#self.kg.mean_death_time + np.int(np.floor(np.random.normal(0, 20, 1)))
                start_time = pick_death_hour - self.predict_window_prior + float(j) * self.time_step_length
                end_time = start_time + self.time_step_length
            else:
                start_time = self.kg.dic_patient[center_node_index]['death_hour'] - self.predict_window_prior + float(
                    j) * self.time_step_length
                end_time = start_time + self.time_step_length
            one_data_vital = self.assign_value_patient(center_node_index, start_time, end_time)
            one_data_lab = self.assign_value_lab(center_node_index, start_time, end_time)
            #one_data_icu_label = self.assign_value_icu_intubation(center_node_index, start_time, end_time)
            # one_data_demo = self.assign_value_demo(center_node_index)
            self.patient_pos_sample_vital[j, 0, :] = one_data_vital
            self.patient_pos_sample_lab[j, 0, :] = one_data_lab
            #self.patient_pos_sample_icu_intubation_label[j,0,:] = one_data_icu_label
            # time_index += 1
        one_data_demo = self.assign_value_demo(center_node_index)
        # one_data_com = self.assign_value_com(center_node_index)
        self.patient_pos_sample_demo[0, :] = one_data_demo
        # self.patient_pos_sample_com[0,:] = one_data_com
        for i in range(self.positive_lab_size):
            index_neighbor = np.int(np.floor(np.random.uniform(0, len(neighbor_patient), 1)))
            patient_id = neighbor_patient[index_neighbor]
            time_seq = self.kg.dic_patient[patient_id]['prior_time_vital'].keys()
            time_seq_int = [np.int(k) for k in time_seq]
            time_seq_int.sort()
            one_data_demo = self.assign_value_demo(patient_id)
            # one_data_com = self.assign_value_com(patient_id)
            self.patient_pos_sample_demo[i + 1, :] = one_data_demo
            # self.patient_pos_sample_com[i+1,:] = one_data_com
            # time_index = 0
            # for j in time_seq_int:
            for j in range(self.time_sequence):
                # if time_index == self.time_sequence:
                #   break
                # self.time_index = np.int(j)
                # start_time = float(j)*self.time_step_length
                # end_time = start_time + self.time_step_length
                if flag == 0:
                    pick_death_hour = self.kg.dic_patient[center_node_index]['pick_time']#self.kg.mean_death_time + np.int(np.floor(np.random.normal(0, 20, 1)))
                    start_time = pick_death_hour - self.predict_window_prior + float(j) * self.time_step_length
                    end_time = start_time + self.time_step_length
                else:
                    start_time = self.kg.dic_patient[patient_id]['death_hour'] - self.predict_window_prior + float(
                        j) * self.time_step_length
                    end_time = start_time + self.time_step_length
                one_data_vital = self.assign_value_patient(patient_id, start_time, end_time)
                one_data_lab = self.assign_value_lab(patient_id, start_time, end_time)
                #one_data_icu_label = self.assign_value_icu_intubation(patient_id, start_time, end_time)
                self.patient_pos_sample_vital[j, i + 1, :] = one_data_vital
                self.patient_pos_sample_lab[j, i + 1, :] = one_data_lab
                #self.patient_pos_sample_icu_intubation_label[j,i+1,:] = one_data_icu_label
                # time_index += 1

    def get_negative_patient(self, center_node_index):
        self.patient_neg_sample_vital = np.zeros((self.time_sequence, self.negative_lab_size, self.item_size))
        self.patient_neg_sample_lab = np.zeros((self.time_sequence, self.negative_lab_size, self.lab_size))
        self.patient_neg_sample_icu_intubation_label = np.zeros((self.time_sequence,self.negative_lab_size,2))
        self.patient_neg_sample_demo = np.zeros((self.negative_lab_size, self.demo_size))
        self.patient_neg_sample_com = np.zeros((self.negative_lab_size, self.com_size))
        if self.kg.dic_patient[center_node_index]['death_flag'] == 0:
            neighbor_patient = self.kg.dic_death[1]
            flag = 1
        else:
            neighbor_patient = self.kg.dic_death[0]
            flag = 0
        for i in range(self.negative_lab_size):
            index_neighbor = np.int(np.floor(np.random.uniform(0, len(neighbor_patient), 1)))
            patient_id = neighbor_patient[index_neighbor]
            time_seq = self.kg.dic_patient[patient_id]['prior_time_vital'].keys()
            time_seq_int = [np.int(k) for k in time_seq]
            time_seq_int.sort()
            time_index = 0
            one_data_demo = self.assign_value_demo(patient_id)
            # one_data_com = self.assign_value_com(patient_id)
            self.patient_neg_sample_demo[i, :] = one_data_demo
            # self.patient_neg_sample_com[i,:] = one_data_com
            # for j in time_seq_int:
            for j in range(self.time_sequence):
                # if time_index == self.time_sequence:
                #   break
                # self.time_index = np.int(j)
                # start_time = float(j)*self.time_step_length
                # end_time = start_time + self.time_step_length
                if flag == 0:
                    pick_death_hour = self.kg.dic_patient[center_node_index]['pick_time']#self.kg.mean_death_time + np.int(np.floor(np.random.normal(0, 20, 1)))
                    start_time = pick_death_hour - self.predict_window_prior + float(j) * self.time_step_length
                    end_time = start_time + self.time_step_length
                else:
                    start_time = self.kg.dic_patient[patient_id]['death_hour'] - self.predict_window_prior + float(
                        j) * self.time_step_length
                    end_time = start_time + self.time_step_length
                one_data_vital = self.assign_value_patient(patient_id, start_time, end_time)
                one_data_lab = self.assign_value_lab(patient_id, start_time, end_time)
                #one_data_icu_label = self.assign_value_icu_intubation(patient_id,start_time,end_time)
                self.patient_neg_sample_vital[j, i, :] = one_data_vital
                self.patient_neg_sample_lab[j, i, :] = one_data_lab
                #self.patient_neg_sample_icu_intubation_label[j,i,:] = one_data_icu_label
                # time_index += 1

    """
    def get_negative_sample_rep(self):
        self.item_neg_sample = np.zeros((self.negative_lab_size,self.item_size))
        index = 0
        for i in self.neg_nodes_item:
            one_sample_neg_item = self.assign_value_item(i)
            self.item_neg_sample[index,:] = one_sample_neg_item
            index += 1
    """

    def SGNN_loss(self):
        """
        implement sgnn loss
        """
        negative_training_norm = tf.math.l2_normalize(self.x_negative, axis=2)

        skip_training = tf.broadcast_to(self.x_origin,
                                        [self.batch_size, 1,#self.negative_sample_size,
                                         self.latent_dim + self.latent_dim_demo])

        skip_training_norm = tf.math.l2_normalize(skip_training, axis=2)

        dot_prod = tf.multiply(skip_training_norm, negative_training_norm)

        dot_prod_sum = tf.reduce_sum(dot_prod, 2)

        sum_log_dot_prod = tf.math.log(tf.math.sigmoid(tf.math.negative(tf.reduce_mean(dot_prod_sum, 1))))

        positive_training = tf.broadcast_to(self.x_origin, [self.batch_size, 1,#self.positive_sample_size,
                                                            self.latent_dim + self.latent_dim_demo])

        positive_skip_norm = tf.math.l2_normalize(self.x_skip, axis=2)

        positive_training_norm = tf.math.l2_normalize(positive_training, axis=2)

        dot_prod_positive = tf.multiply(positive_skip_norm, positive_training_norm)

        dot_prod_sum_positive = tf.reduce_sum(dot_prod_positive, 2)

        sum_log_dot_prod_positive = tf.math.log(tf.math.sigmoid(tf.reduce_mean(dot_prod_sum_positive, 1)))

        self.negative_sum = tf.math.negative(
            tf.reduce_sum(tf.math.add(sum_log_dot_prod, sum_log_dot_prod_positive)))

    def SGNN_loss_contrast(self):
        """
        mplement sgnn loss contrast
        """
        negative_training_norm = tf.math.l2_normalize(self.relation_patients_neg_add_weight, axis=2)

        skip_training = tf.broadcast_to(self.x_origin_contrast_,
                                        [self.batch_size, self.negative_sample_size,self.lab_size+self.item_size,
                                         self.latent_dim + self.latent_dim_demo])

        skip_training_norm = tf.math.l2_normalize(skip_training, axis=2)

        dot_prod = tf.multiply(skip_training_norm, negative_training_norm)

        dot_prod_sum = tf.reduce_sum(dot_prod, 2)

        sum_log_dot_prod = tf.math.log(tf.math.sigmoid(tf.math.negative(tf.reduce_mean(dot_prod_sum, 1))))

        positive_training = tf.broadcast_to(self.x_origin_contrast, [self.batch_size, self.positive_sample_size,self.lab_size+self.item_size,
                                                            self.latent_dim + self.latent_dim_demo])

        positive_skip_norm = tf.math.l2_normalize(self.relation_patients_pos_add_weight, axis=2)

        positive_training_norm = tf.math.l2_normalize(positive_training, axis=2)

        dot_prod_positive = tf.multiply(positive_skip_norm, positive_training_norm)

        dot_prod_sum_positive = tf.reduce_sum(dot_prod_positive, 2)

        sum_log_dot_prod_positive = tf.math.log(tf.math.sigmoid(tf.reduce_mean(dot_prod_sum_positive, 1)))

        self.negative_sum_contrast = tf.math.negative(
            tf.reduce_sum(tf.math.add(sum_log_dot_prod, sum_log_dot_prod_positive)))


    def SGNN_loss_contrast_multi(self):
        """
        mplement sgnn loss contrast
        """
        negative_training_norm = tf.math.l2_normalize(self.x_negative_contrast, axis=2)

        skip_training = tf.broadcast_to(self.x_origin_contrast,
                                        [self.batch_size, self.negative_sample_size,self.item_size+self.lab_size,
                                         self.latent_dim + self.latent_dim_demo])

        skip_training_norm = tf.math.l2_normalize(skip_training, axis=2)

        dot_prod = tf.multiply(skip_training_norm, negative_training_norm)

        dot_prod_sum = tf.reduce_sum(dot_prod, 2)

        sum_log_dot_prod = tf.math.log(tf.math.sigmoid(tf.math.negative(tf.reduce_mean(dot_prod_sum, 1))))

        positive_training = tf.broadcast_to(self.x_origin_contrast, [self.batch_size, self.positive_sample_size,self.item_size+self.lab_size,
                                                            self.latent_dim + self.latent_dim_demo])

        positive_skip_norm = tf.math.l2_normalize(self.x_skip_contrast, axis=2)

        positive_training_norm = tf.math.l2_normalize(positive_training, axis=2)

        dot_prod_positive = tf.multiply(positive_skip_norm, positive_training_norm)

        dot_prod_sum_positive = tf.reduce_sum(dot_prod_positive, 2)

        sum_log_dot_prod_positive = tf.math.log(tf.math.sigmoid(tf.reduce_mean(dot_prod_sum_positive, 1)))

        self.negative_sum_contrast = tf.math.negative(
            tf.reduce_sum(tf.math.add(sum_log_dot_prod, sum_log_dot_prod_positive)))

    def config_model(self):
        self.lstm_cell()
        self.demo_layer()
        # self.softmax_loss()
        self.build_dhgm_model()
        self.get_latent_rep_hetero()

        self.SGNN_loss()
        self.SGNN_loss_contrast()
        #self.train_step_neg = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.negative_sum)
        self.train_step_neg = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(0.8*self.negative_sum+0.2*self.negative_sum_contrast)
         #self.train_step_cross_entropy = tf.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()



    def assign_value_patient(self, patientid, start_time, end_time):
        self.one_sample = np.zeros(self.item_size)
        self.freq_sample = np.zeros(self.item_size)
        self.times = []
        for i in self.kg.dic_patient[patientid]['prior_time_vital'].keys():
            if (np.int(i) > start_time or np.int(i) == start_time) and np.int(i) < end_time:
                self.times.append(i)
        for j in self.times:
            for i in self.kg.dic_patient[patientid]['prior_time_vital'][str(j)].keys():
                mean = np.float(self.kg.dic_vital[i]['mean_value'])
                std = np.float(self.kg.dic_vital[i]['std'])
                ave_value = np.mean(
                    [np.float(k) for k in self.kg.dic_patient[patientid]['prior_time_vital'][str(j)][i]])
                index = self.kg.dic_vital[i]['index']
                if std == 0:
                    self.one_sample[index] += 0
                    self.freq_sample[index] += 1
                else:
                    self.one_sample[index] = (np.float(ave_value) - mean) / std
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
        self.freq_sample_lab = np.zeros(self.lab_size)
        self.times_lab = []
        for i in self.kg.dic_patient[patientid]['prior_time_lab'].keys():
            if (np.int(i) > start_time or np.int(i) == start_time) and np.int(i) < end_time:
                self.times_lab.append(i)
        for j in self.times_lab:
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
                ave_value = np.mean([np.float(k) for k in self.kg.dic_patient[patientid]['prior_time_lab'][str(j)][i]])
                index = self.kg.dic_lab[i]['index']
                if std == 0:
                    self.one_sample_lab[index] += 0
                    self.freq_sample_lab[index] += 1
                else:
                    self.one_sample_lab[index] += (np.float(ave_value) - mean) / std
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

    def compute_time_seq_single(self,central_node_variable):
        """
        compute single node feature values
        """
        time_seq_variable = np.zeros((self.item_size + self.lab_size, self.time_sequence))
        if self.kg.dic_patient[central_node_variable]['death_flag'] == 0:
            flag = 0
            # neighbor_patient = self.kg.dic_death[0]
        else:
            flag = 1
            # neighbor_patient = self.kg.dic_death[1]
        time_seq = self.kg.dic_patient[central_node_variable]['prior_time_vital'].keys()
        time_seq_int = [np.int(k) for k in time_seq]
        time_seq_int.sort()
        # time_index = 0
        # for j in self.time_seq_int:
        for j in range(self.time_sequence):
            # if time_index == self.time_sequence:
            #    break
            if flag == 0:
                pick_death_hour = self.kg.dic_patient[central_node_variable][
                    'pick_time']  # self.kg.mean_death_time + np.int(np.floor(np.random.normal(0, 20, 1)))
                start_time = pick_death_hour - self.predict_window_prior + float(j) * self.time_step_length
                end_time = start_time + self.time_step_length
            else:
                start_time = self.kg.dic_patient[central_node_variable][
                                 'death_hour'] - self.predict_window_prior + float(
                    j) * self.time_step_length
                end_time = start_time + self.time_step_length
            one_data_vital = self.assign_value_patient(central_node_variable, start_time, end_time)
            one_data_lab = self.assign_value_lab(central_node_variable, start_time, end_time)
            # one_data_icu_label = self.assign_value_icu_intubation(center_node_index, start_time, end_time)
            # one_data_demo = self.assign_value_demo(center_node_index)
            # self.patient_pos_sample_vital[j, 0, :] = one_data_vital
            # self.patient_pos_sample_lab[j, 0, :] = one_data_lab
            one_data = np.concatenate([one_data_vital, one_data_lab])
            time_seq_variable[:,j] = one_data

        return time_seq_variable

    def compute_relation_indicator(self,central_node,context_node):
        softmax_weight = np.zeros((self.item_size+self.lab_size))
        #features = list(self.kg.dic_vital.keys())+list(self.kg.dic_lab.keys())
        center_data = np.mean(self.compute_time_seq_single(central_node),axis=1)
        context_data = np.mean(self.compute_time_seq_single(context_node),axis=1)
        difference = np.abs(center_data-context_data)
        for i in range(self.item_size+self.lab_size):
            if difference[i] < self.softmax_weight_threshold:
                softmax_weight[i] = 1

        return softmax_weight

    def compute_relation_indicator_neg(self,central_node,context_node):
        softmax_weight = np.zeros((self.item_size+self.lab_size))
        #features = list(self.kg.dic_vital.keys())+list(self.kg.dic_lab.keys())
        center_data = np.mean(self.compute_time_seq_single(central_node),axis=1)
        context_data = np.mean(self.compute_time_seq_single(context_node),axis=1)
        difference = np.abs(center_data-context_data)
        for i in range(self.item_size+self.lab_size):
            if difference[i] > self.softmax_weight_threshold:
                softmax_weight[i] = 1

        return softmax_weight

    def compute_soft_weight(self):
        soft_weight = np.zeros((self.positive_lab_size+self.negative_lab_size, self.item_size+self.lab_size))
        self.variable_pos = np.mean(np.concatenate((self.patient_pos_sample_vital,self.patient_pos_sample_lab),axis=2),axis=0)
        self.variable_neg = np.mean(np.concatenate((self.patient_neg_sample_vital,self.patient_neg_sample_lab),axis=2),axis=0)
        self.variable_compare = np.concatenate((self.variable_pos,self.variable_neg),axis=0)
        self.variable_origin = self.variable_pos[0,:]
        for i in range(self.positive_lab_size):
            pos_compare = self.variable_compare[1+i,:]
            compare = np.abs(self.variable_origin-pos_compare)
            for j in range(self.item_size+self.lab_size):
                if compare[j] < self.softmax_weight_threshold:
                    soft_weight[i,j] = 1

        for i in range(self.negative_lab_size):
            pos_compare = self.variable_compare[self.positive_lab_size+i,:]
            compare = np.abs(self.variable_origin-pos_compare)
            for j in range(self.item_size+self.lab_size):
                if compare[j] > self.softmax_weight_threshold:
                    soft_weight[i+self.positive_lab_size,j] = 1

        return soft_weight



    def get_batch_train(self, data_length, start_index, data):
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
        train_one_batch_soft_weight = np.zeros(
        (data_length,self.positive_lab_size + self.negative_lab_size, self.item_size+self.lab_size))
        # train_one_batch_item = np.zeros((data_length,self.positive_lab_size+self.negative_lab_size,self.item_size))
        train_one_batch_mortality = np.zeros((data_length, 2, 2))
        one_batch_logit = np.zeros((data_length, 2))
        self.real_logit = np.zeros(data_length)
        # self.item_neg_sample = np.zeros((self.negative_lab_size, self.item_size))
        # self.item_pos_sample = np.zeros((self.positive_lab_size, self.item_size))
        index_batch = 0
        index_increase = 0
        # while index_batch < data_length:
        for i in range(data_length):
            self.check_patient = i
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
            else:
                train_one_batch_mortality[i, 0, :] = [0, 1]
                train_one_batch_mortality[i, 1, :] = [1, 0]
                one_batch_logit[i, 1] = 1

            self.get_positive_patient(self.patient_id)
            self.get_negative_patient(self.patient_id)
            self.train_one_data_soft_weight = self.compute_soft_weight()
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
            train_one_batch_soft_weight[i,:,:] = self.train_one_data_soft_weight

        return train_one_batch_vital, train_one_batch_lab, train_one_batch_demo, one_batch_logit, train_one_batch_mortality, train_one_batch_com,train_one_batch_icu_intubation,train_one_batch_soft_weight


    def train(self):
        """
        train the system
        """
        self.length_train = len(self.train_data)
        init_hidden_state = np.zeros(
            (self.batch_size, 1 + self.positive_lab_size + self.negative_lab_size, self.latent_dim))
        iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))

        for j in range(self.epoch):
            print('epoch')
            print(j)
            for i in range(iteration):
                self.train_one_batch_vital, self.train_one_batch_lab, self.train_one_batch_demo, self.one_batch_logit, self.one_batch_mortality, self.one_batch_com,self.one_batch_icu_intubation,self.one_batch_soft_weight = self.get_batch_train(
                    self.batch_size, i * self.batch_size, self.train_data)

                self.err_ = self.sess.run([self.negative_sum, self.train_step_neg],
                                          feed_dict={self.input_x_vital: self.train_one_batch_vital,
                                                     self.input_x_lab: self.train_one_batch_lab,
                                                     self.input_x_demo: self.train_one_batch_demo,
                                                     self.relation_patients: self.orthog_input,
                                                     # self.input_x_com: self.one_batch_com,
                                                     # self.lab_test: self.one_batch_item,
                                                     self.mortality: self.one_batch_mortality,
                                                     self.init_hiddenstate: init_hidden_state,
                                                     self.input_icu_intubation:self.one_batch_icu_intubation,
                                                     self.relation_weight:self.one_batch_soft_weight})
                print(self.err_[0])

                """
                self.err_lstm = self.sess.run([self.cross_entropy, self.train_step_cross_entropy,self.init_hiddenstate,self.output_layer,self.logit_sig],
                                     feed_dict={self.input_x: self.train_one_batch,
                                                self.input_y_logit: self.one_batch_logit,
                                                self.init_hiddenstate:init_hidden_state})
                print(self.err_lstm[0])
                """

    def test(self, data):
        Death = np.zeros([1,2])
        Death[0][1] = 1
        test_length = len(data)
        init_hidden_state = np.zeros(
            (test_length, 1 + self.positive_lab_size + self.negative_lab_size, self.latent_dim))
        self.test_data_batch_vital, self.test_one_batch_lab, self.test_one_batch_demo, self.test_logit, self.test_mortality, self.test_com,self.one_batch_icu_intubation,self.one_batch_soft_weight = self.get_batch_train(
            test_length, 0, data)
        self.test_patient = self.sess.run(self.Dense_patient, feed_dict={self.input_x_vital: self.test_data_batch_vital,
                                                                         self.input_x_lab: self.test_one_batch_lab,
                                                                         self.input_x_demo: self.test_one_batch_demo,
                                                                         # self.input_x_com: self.test_com,
                                                                         self.init_hiddenstate: init_hidden_state,
                                                                         self.input_icu_intubation:self.one_batch_icu_intubation})[:,
                            0, :]
        """
        self.test_att_score = self.sess.run([self.score_attention,self.input_importance,self.input_x],feed_dict={self.input_x_vital: self.test_data_batch_vital,
                                                                         self.input_x_lab: self.test_one_batch_lab,
                                                                         self.input_x_demo: self.test_one_batch_demo,
                                                                         self.init_hiddenstate: init_hidden_state,
                                                                         self.Death_input: Death,
                                                                         self.input_icu_intubation:self.one_batch_icu_intubation})
        """


        single_mortality = np.zeros((1, 2, 2))
        single_mortality[0][0][0] = 1
        single_mortality[0][1][1] = 1
        self.mortality_test = self.sess.run(self.Dense_mortality, feed_dict={self.mortality: single_mortality})[0]
        self.score = np.zeros(test_length)
        for i in range(test_length):
            embed_single_patient = self.test_patient[i] / np.linalg.norm(self.test_patient[i])
            embed_mortality = self.mortality_test[1] / np.linalg.norm(self.mortality_test[1])
            self.score[i] = np.matmul(embed_single_patient, embed_mortality.T)

        self.correct = 0
        self.tp_correct = 0
        self.tp_neg = 0
        self.correct_predict_death = []
        for i in range(test_length):
            if self.test_logit[i, 1] == 1:
                self.tp_correct += 1
            if self.test_logit[i, 0] == 1:
                self.tp_neg += 1
            if self.score[i] < 0 and self.test_logit[i, 0] == 1:
                self.correct += 1
            if self.score[i] > 0 and self.test_logit[i, 1] == 1:
                self.correct_predict_death.append(i)
                self.correct += 1

        self.acc = np.float(self.correct) / test_length
        """
        self.correct_predict_death = np.array(self.correct_predict_death)

        feature_len = self.item_size + self.lab_size


        self.test_data_scores = self.test_att_score[1][self.correct_predict_death,:,0,:]
        self.ave_data_scores = np.zeros((self.time_sequence,feature_len))

        count = 0
        value = 0

        for j in range(self.time_sequence):
            for p in range(feature_len):
                for i in range(self.correct_predict_death.shape[0]):
                    if self.test_data_scores[i,j,p]!=0:
                        count += 1
                        value += self.test_data_scores[i,j,p]
                if count == 0:
                    continue
                self.ave_data_scores[j,p] = float(value/count)
                count = 0
                value = 0
        """


        """
        self.tp_test = 0
        self.fp_test = 0
        self.fn_test = 0
        for i in range(test_length):
            if self.score[i] > 0 and self.test_logit[i, 1] == 1:
                self.tp_test += 1
            if self.score[i] < 0 and self.test_logit[i, 1] == 1:
                self.fn_test += 1
            if self.score[i] > 0 and self.test_logit[i, 0] == 1:
                self.fp_test += 1

        self.precision_test = np.float(self.tp_test) / (self.tp_test + self.fp_test)
        self.recall_test = np.float(self.tp_test) / (self.tp_test + self.fn_test)
        self.f1_test = 2 * (self.precision_test * self.recall_test) / (self.precision_test + self.recall_test)
        """

        threshold = -1.01
        self.resolution = 0.01
        self.tp_total = []
        self.fp_total = []
        self.precision_total = []
        self.recall_total = []

        while (threshold < 1.01):
            tp_test = 0
            fp_test = 0
            fn_test = 0
            for i in range(test_length):
                if self.test_logit[i, 1] == 1 and self.score[i] > threshold:
                    tp_test += 1
                if self.test_logit[i, 0] == 1 and self.score[i] > threshold:
                    fp_test += 1
                if self.score[i] < threshold and self.test_logit[i, 1] == 1:
                    fn_test += 1

            tp_rate = tp_test / self.tp_correct
            fp_rate = fp_test / self.tp_neg
            if (tp_test+fp_test) == 0:
                precision_test = 1.0
            else:
                precision_test = np.float(tp_test) / (tp_test + fp_test)
            recall_test = np.float(tp_test) / (tp_test + fn_test)
            self.tp_total.append(tp_rate)
            self.fp_total.append(fp_rate)
            self.precision_total.append(precision_test)
            self.recall_total.append(recall_test)
            threshold += self.resolution

    def generate_orthogonal_relatoin(self):
        self.orthog_init = tf.keras.initializers.Orthogonal()
        self.orthog_value = self.orthog_init(shape=self.shape_relation_patient)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        self.orthog_input = self.sess.run(self.orthog_value)
        self.orthog_input = np.expand_dims(self.orthog_input,0)
        self.orthog_input = np.expand_dims(self.orthog_input,0)
        self.sess.close()


    def cross_validation(self):
        self.f1_score_total = []
        self.acc_total = []
        self.area_total = []
        self.auprc_total = []
        self.test_logit_total = []
        self.tp_score_total = []
        self.fp_score_total = []
        self.precision_score_total = []
        self.precision_curve_total = []
        self.recall_score_total = []
        self.recall_curve_total = []
        self.test_patient_whole = []
        feature_len = self.item_size + self.lab_size
        #self.ave_data_scores_total = np.zeros((self.time_sequence, feature_len))
        self.generate_orthogonal_relatoin()


        self.config_model()
        for i in range(3):
            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            self.train_data = self.train_data_whole[i]
            self.test_data = self.test_data_whole[i]
            self.train()
            self.test(self.test_data)
            #self.f1_score_total.append(self.f1_test)
            #self.acc_total.append(self.acc)
            self.tp_score_total.append(self.tp_total)
            self.fp_score_total.append(self.fp_total)
            self.cal_auc()
            self.cal_auprc()
            self.area_total.append(self.area)
            self.auprc_total.append(self.area_auprc)
            #self.precision_score_total.append(self.precision_test)
            #self.recall_score_total.append(self.recall_test)
            #self.precision_curve_total.append(self.precision_total)
            #self.recall_curve_total.append(self.recall_total)
            self.test_patient_whole.append(self.test_patient)
            self.test_logit_total.append(self.test_logit)
            #self.ave_data_scores_total += self.ave_data_scores
            self.sess.close()

        #self.ave_data_scores_total = self.ave_data_scores_total/5
        #self.norm = np.linalg.norm(self.ave_data_scores_total)
        #self.ave_data_scores_total = self.ave_data_scores_total/self.norm
        self.tp_ave_score = np.sum(self.tp_score_total,0)/5
        self.fp_ave_score = np.sum(self.fp_score_total,0)/5
        self.precision_ave_score = np.sum(self.precision_curve_total,0)/5
        self.recall_ave_score = np.sum(self.recall_curve_total,0)/5
        #print("f1_ave_score")
        #print(np.mean(self.f1_score_total))
        #print("acc_ave_score")
        #print(np.mean(self.acc_total))
        print("area_ave_score")
        print(np.mean(self.area_total))
        #print("precision_ave_score")
        #print(np.mean(self.precision_total))
        #print("recall_ave_score")
        #print(np.mean(self.recall_total))
        print("auprc_ave_score")
        print(np.mean(self.auprc_total))

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