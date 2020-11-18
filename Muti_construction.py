import numpy as np
import random
import math
import time
import pandas as pd
from scipy.stats import iqr

class Multiplex_network():
    """
    Construct Multiplex Network
    """
    def __init__(self, kg):
        self.kg = kg
        self.time_seq_length = 4
        self.time_step_length = 6
        self.predict_window_prior = self.time_seq_length * self.time_step_length
        pick_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 19,
                    20, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 36, 37, 38, 41, 43,
                    45, 46, 47, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 66,
                    67, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 83]
        pick_num = np.array(pick_num)
        self.feature = list(np.array(list(self.kg.dic_vital.keys()) + list(self.kg.dic_lab.keys()))[pick_num])
        self.feature_length = len(pick_num)
        self.data_length = len(list(self.kg.dic_patient.keys()))
        self.item_size = len(list(kg.dic_vital.keys()))
        self.demo_size = len(list(kg.dic_race.keys()))
        self.lab_size = len(list(kg.dic_lab.keys()))

    def compute_time_seq(self):
        self.time_seq_variable = np.zeros((self.data_length,self.item_size+self.lab_size,self.time_seq_length))
        self.time_seq_index = []
        self.time_seq_variable_name = []
        #count = 0
        variables = list(self.kg.dic_patient.keys())
        for i in range(self.data_length):
            self.time_seq_index.append(i)
            central_node_variable = variables[i]
            self.time_seq_variable_name.append(central_node_variable)
            if self.kg.dic_patient[i]['death_flag'] == 0:
                flag = 0
                #neighbor_patient = self.kg.dic_death[0]
            else:
                flag = 1
                #neighbor_patient = self.kg.dic_death[1]
            time_seq = self.kg.dic_patient[central_node_variable]['prior_time_vital'].keys()
            time_seq_int = [np.int(k) for k in time_seq]
            time_seq_int.sort()
            # time_index = 0
            # for j in self.time_seq_int:
            for j in range(self.time_seq_length):
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
                #self.patient_pos_sample_vital[j, 0, :] = one_data_vital
                #self.patient_pos_sample_lab[j, 0, :] = one_data_lab
                one_data = np.concatenate([one_data_vital,one_data_lab])
                self.time_seq_variable[count,:,j] = one_data

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
