import numpy as np
import random
import math
import time
import pandas as pd
from scipy.stats import iqr
import json
#from LSTM import LSTM_model
from Data_process import kg_process_data
from Dynamic_hgm_death_whole import dynamic_hgm
from Muti_construction import Multiplex_network
from Contrastive_regular import con_regular
from knn_cl import knn_cl
#from MLP import MLP_model


class Kg_construct_ehr():
    """
    construct knowledge graph out of EHR data
    """

    def __init__(self):
        file_path = '/datadrive/tingyi_wanyan/user_tingyi.wanyan/tensorflow_venv/registry_2020-06-29'
        self.reg = file_path + '/registry.csv'
        self.covid_lab = file_path + '/covid19LabTest.csv'
        self.lab = file_path + '/Lab.csv'
        self.vital = file_path + '/vitals.csv'
        file_path_ = '/home/tingyi.wanyan'
        self.lab_comb = 'lab_mapping_comb.csv'
        self.file_path_comorbidity = '/home/tingyi.wanyan/comorbidity_matrix_20200710.csv'

    def read_csv(self):
        self.registry = pd.read_csv(self.reg)
        self.covid_labtest = pd.read_csv(self.covid_lab)
        self.labtest = pd.read_csv(self.lab)
        self.vital_sign = pd.read_csv(self.vital)
        # self.comorbidity = pd.read_csv(self.file_path_comorbidity)
        self.lab_comb = pd.read_csv(self.lab_comb)
        self.reg_ar = np.array(self.registry)
        self.covid_ar = np.array(self.covid_labtest)
        self.labtest_ar = np.array(self.labtest)
        self.vital_sign_ar = np.array(self.vital_sign)
        self.lab_comb_ar = np.array(self.lab_comb)

    def create_kg_dic(self):
        self.dic_patient = {}
        self.dic_vital = {}
        self.dic_lab = {}
        self.dic_filter_patient = {}
        self.dic_lab_category = {}
        self.dic_demographic = {}
        self.dic_race = {}
        self.changed_death = []
        self.crucial_vital = ['CAC - BLOOD PRESSURE', 'CAC - TEMPERATURE', 'CAC - PULSE OXIMETRY',
                              'CAC - RESPIRATIONS', 'CAC - PULSE', 'CAC - HEIGHT', 'CAC - WEIGHT/SCALE']
        index_keep = np.where(self.lab_comb_ar[:, -1] == 1)[0]
        self.lab_comb_keep = self.lab_comb_ar[index_keep]
        index_name = np.where(self.lab_comb_keep[:, -2] == self.lab_comb_keep[:, -2])[0]
        self.lab_test_feature = []
        [self.lab_test_feature.append(i) for i in self.lab_comb_keep[:, -2] if i not in self.lab_test_feature]
        self.lab_comb_keep_ = self.lab_comb_keep[index_name]
        self.cat_comb = self.lab_comb_keep[:, [0, -2]]
        """
        create inital lab dictionary
        """
        index_lab = 0
        for i in range(index_name.shape[0]):
            name_test = self.lab_comb_keep[i][0]
            name_category = self.lab_comb_keep[i][-2]
            if name_test not in self.dic_lab_category.keys():
                self.dic_lab_category[name_test] = name_category
                if name_category not in self.dic_lab:
                    self.dic_lab[name_category] = {}
                    # self.dic_lab[name_category]['patient_values'] = {}
                    # self.dic_lan[name_category]['specific name']={}
                    # self.dic_lab[name_category].setdefault('specific_name',[]).append(name_test)
                    self.dic_lab[name_category]['index'] = index_lab
                    index_lab += 1
                # else:
                #   self.dic_lab[name_category].setdefault('specific_name',[]).append(name_test)
        """
        create initial vital sign dictionary
        """
        index_vital = 0
        for i in self.crucial_vital:
            if i == 'CAC - BLOOD PRESSURE':
                self.dic_vital['high'] = {}
                self.dic_vital['high']['index'] = index_vital
                index_vital += 1
                self.dic_vital['low'] = {}
                self.dic_vital['low']['index'] = index_vital
                index_vital += 1
            else:
                self.dic_vital[i] = {}
                self.dic_vital[i]['index'] = index_vital
                index_vital += 1

        """
        get all patient with admit time
        """
        admit_time = np.where(self.reg_ar[:,1]==self.reg_ar[:,1])[0]
        self.admit = self.reg_ar[admit_time,:]
        covid_obv = np.where(self.admit[:,8]==self.admit[:,8])[0]
        self.covid_ar = self.admit[covid_obv,:]

        """
        filter out the first visit ID
        """
        for i in range(self.covid_ar.shape[0]):
            print("im here in filter visit ID")
            print(i)
            mrn_single = self.covid_ar[i,45]
            visit_id = self.covid_ar[i,65]
            if visit_id == visit_id:
                if mrn_single not in self.dic_patient.keys():
                    self.dic_patient[mrn_single] = {}
                    self.dic_patient[mrn_single]['prior_time_vital'] = {}
                    self.dic_patient[mrn_single]['prior_time_lab'] = {}
                in_admit_time_single = self.covid_ar[i,1]

                self.in_admit_time = in_admit_time_single.split(' ')
                in_admit_date = [np.int(j) for j in self.in_admit_time[0].split('-')]
                in_admit_date_value = (in_admit_date[0] * 365.0 + in_admit_date[1] * 30 + in_admit_date[2]) * 24 * 60
                self.in_admit_time_ = [np.int(j) for j in self.in_admit_time[1].split(':')[0:-1]]
                in_admit_time_value = self.in_admit_time_[0] * 60.0 + self.in_admit_time_[1]
                total_in_admit_time_value = in_admit_date_value + in_admit_time_value
                self.dic_patient[mrn_single].setdefault('Admit_time_values', []).append(total_in_admit_time_value)
                """
                filter intubation
                """
                if self.covid_ar[i, 29] == self.covid_ar[i, 29]:
                    self.dic_patient[mrn_single]['icu_label'] = 1
                    in_time_single = self.covid_ar[i, 29]
                    self.in_time = in_time_single.split(' ')
                    in_date = [np.int(j) for j in self.in_time[0].split('-')]
                    in_date_value = (in_date[0] * 365.0 + in_date[1] * 30 + in_date[2]) * 24 * 60
                    self.in_time_ = [np.int(j) for j in self.in_time[1].split(':')[0:-1]]
                    in_time_value = self.in_time_[0] * 60.0 + self.in_time_[1]
                    total_in_time_value = in_date_value + in_time_value
                    self.dic_patient[mrn_single]['in_icu_time'] = self.in_time
                    self.dic_patient[mrn_single]['in_date'] = in_date
                    self.dic_patient[mrn_single]['in_time'] = self.in_time_
                    self.dic_patient[mrn_single]['total_in_icu_time_value'] = total_in_time_value
                else:
                    self.dic_patient[mrn_single]['icu_label'] = 0
                """
                filter intubation
                """
                if self.covid_ar[i, 35] == self.covid_ar[i, 35]:
                    self.dic_patient[mrn_single]['intubation_label'] = 1
                    in_time_single = self.covid_ar[i, 35]
                    self.in_time = in_time_single.split(' ')
                    in_date = [np.int(i) for i in self.in_time[0].split('-')]
                    in_date_value = (in_date[0] * 365.0 + in_date[1] * 30 + in_date[2]) * 24 * 60
                    self.in_time_ = [np.int(i) for i in self.in_time[1].split(':')[0:-1]]
                    in_time_value = self.in_time_[0] * 60.0 + self.in_time_[1]
                    total_in_time_value = in_date_value + in_time_value
                    self.dic_patient[mrn_single]['intubation_time'] = self.in_time
                    self.dic_patient[mrn_single]['intubation_date'] = in_date
                    self.dic_patient[mrn_single]['intubation_time'] = self.in_time_
                    self.dic_patient[mrn_single]['total_intubation_time_value'] = total_in_time_value
                else:
                    self.dic_patient[mrn_single]['intubation_label'] = 0

                """
                filter mortality
                """
                if self.covid_ar[i, 11] == self.covid_ar[i, 11]:
                    death_flag = 1
                    death_time_ = kg.covid_ar[i][11]
                    self.dic_patient[mrn_single]['death_time'] = death_time_
                    death_time = death_time_.split(' ')
                    death_date = [np.int(l) for l in death_time[0].split('-')]
                    death_date_value = (death_date[0] * 365.0 + death_date[1] * 30 + death_date[2]) * 24 * 60
                    dead_time_ = [np.int(l) for l in death_time[1].split(':')[0:-1]]
                    dead_time_value = dead_time_[0] * 60.0 + dead_time_[1]
                    total_dead_time_value = death_date_value + dead_time_value
                    self.dic_patient[mrn_single]['death_value'] = total_dead_time_value
                else:
                    death_flag = 0
                self.dic_patient[mrn_single]['death_flag'] = death_flag

        """
        change EXP and 20 to death
        """
        for i in range(self.covid_ar.shape[0]):
            if self.covid_ar[i,13]=='EXP' or self.covid_ar[i,13]=='20':
                mrn_single = self.covid_ar[i, 45]
                if self.covid_ar[i][12] == self.covid_ar[i][12]:
                    discharge_time_ = self.covid_ar[i][12]
                    self.dic_patient[mrn_single]['discharge_time'] = discharge_time_
                    discharge_time = discharge_time_.split(' ')
                    discharge_date = [np.int(l) for l in discharge_time[0].split('-')]
                    discharge_date_value = (discharge_date[0] * 365.0 + discharge_date[1] * 30 + discharge_date[2]) * 24 * 60
                    dischar_time_ = [np.int(l) for l in discharge_time[1].split(':')[0:-1]]
                    discharge_time_value = dischar_time_[0] * 60.0 + dischar_time_[1]
                    total_discharge_time_value = discharge_date_value + discharge_time_value
                    self.dic_patient[mrn_single]['discharge_value'] = total_discharge_time_value
                    if self.dic_patient[mrn_single]['death_flag'] == 0:
                        self.changed_death.append(mrn_single)
                        self.dic_patient[mrn_single]['death_flag'] = 1
                        self.dic_patient[mrn_single]['death_value'] = total_discharge_time_value


        """
        filter out labels
        """
        self.total_in_icu_time = []
        self.total_intubation_time = []
        self.total_death_time = []
        self.dic_death = {}
        self.dic_intubation = {}
        self.dic_in_icu = {}
        for i in self.dic_patient.keys():
            self.dic_patient[i]['Admit_time_values'] = np.sort(self.dic_patient[i]['Admit_time_values'])
            if self.dic_patient[i]['icu_label'] == 1:
                if len(self.dic_patient[i]['Admit_time_values'])>1:
                    if self.dic_patient[i]['total_in_icu_time_value']>self.dic_patient[i]['Admit_time_values'][1]:
                        self.dic_patient[i]['icu_label'] = 0
                        self.dic_patient[i]['filter_first_icu_visit'] = 1
            if self.dic_patient[i]['death_flag'] == 1:
                if len(self.dic_patient[i]['Admit_time_values'])>1:
                    if self.dic_patient[i]['death_value']>self.dic_patient[i]['Admit_time_values'][1]:
                        self.dic_patient[i]['death_flag'] = 0
                        self.dic_patient[i]['filter_first_death_visit'] = 1

            if self.dic_patient[i]['intubation_label'] == 1:
                if len(self.dic_patient[i]['Admit_time_values'])>1:
                    if self.dic_patient[i]['total_intubation_time_value']>self.dic_patient[i]['Admit_time_values'][1]:
                        self.dic_patient[i]['intubation_label'] = 0
                        self.dic_patient[i]['filter_first_intubation_visit'] = 1

        for i in self.dic_patient.keys():
            if self.dic_patient[i]['icu_label'] == 1:
                total_in_icu_time_value = self.dic_patient[i]['total_in_icu_time_value']
                total_in_admit_time_value = self.dic_patient[i]['Admit_time_values'][0]
                self.dic_patient[i]['in_icu_hour'] = np.int(
                    np.floor((total_in_icu_time_value - total_in_admit_time_value) / 60))
                self.total_in_icu_time.append(kg.dic_patient[i]['in_icu_hour'])
                self.dic_in_icu.setdefault(1, []).append(i)
            if self.dic_patient[i]['icu_label'] == 0:
                self.dic_in_icu.setdefault(0, []).append(i)

            if self.dic_patient[i]['death_flag'] == 1:
                total_death_value = self.dic_patient[i]['death_value']
                self.dic_patient[i]['death_hour'] = np.int(
                    np.floor((total_death_value - self.dic_patient[i]['Admit_time_values'][0]) / 60))
                self.total_death_time.append(self.dic_patient[i]['death_hour'])
                self.dic_death.setdefault(1, []).append(i)
            if self.dic_patient[i]['death_flag'] == 0:
                self.dic_death.setdefault(0, []).append(i)
            if self.dic_patient[i]['intubation_label'] == 1:
                total_intubation_time_value = self.dic_patient[i]['total_intubation_time_value']
                total_in_admit_time_value = self.dic_patient[i]['Admit_time_values'][0]
                self.dic_patient[i]['intubation_hour'] = np.int(
                    np.floor((total_intubation_time_value - total_in_admit_time_value) / 60))
                self.total_intubation_time.append(self.dic_patient[i]['intubation_hour'])
                self.dic_intubation.setdefault(1, []).append(i)
            if self.dic_patient[i]['intubation_label'] == 0:
                self.dic_intubation.setdefault(0, []).append(i)


        self.total_data_mortality = []
        self.un_correct_mortality = []
        self.total_data_intubation = []
        self.un_correct_intubation = []
        self.total_data_icu = []
        self.un_correct_icu = []

        for i in self.dic_patient.keys():
            if self.dic_patient[i]['death_flag'] == 0:
                self.total_data_mortality.append(i)
            if self.dic_patient[i]['death_flag'] == 1:
                if self.dic_patient[i]['death_hour'] > 0:
                    self.total_data_mortality.append(i)
                else:
                    self.un_correct_mortality.append(i)
            if self.dic_patient[i]['intubation_label'] == 0:
                self.total_data_intubation.append(i)
            if self.dic_patient[i]['intubation_label'] == 1:
                if self.dic_patient[i]['intubation_hour'] > 0:
                    self.total_data_intubation.append(i)
                else:
                    self.un_correct_intubation.append(i)
            if self.dic_patient[i]['icu_label'] == 0:
                self.total_data_icu.append(i)
            if self.dic_patient[i]['icu_label'] == 1:
                if self.dic_patient[i]['in_icu_hour'] > 0:
                    self.total_data_icu.append(i)
                else:
                    self.un_correct_icu.append(i)

        index_race = 0
        for i in self.dic_patient.keys():
            index_race_ = np.where(self.covid_ar[:, 45] == i)[0]
            self.check_index = index_race_
            race = 0
            for j in index_race_:
                race_check = self.covid_ar[:, 61][j]
                if race_check == race_check:
                    race = race_check
                    break
            for j in index_race_:
                age_check = self.covid_ar[:, 7][j]
                if age_check == age_check:
                    age = age_check
                    break
            for j in index_race_:
                gender_check = self.covid_ar[:, 24][j]
                if gender_check == gender_check:
                    gender = gender_check
                    break
            # self.dic_race['Age']=age
            # self.dic_race['gender']=gender
            if race == 0:
                continue
            if race[0] == 'A':
                if 'A' not in self.dic_race:
                    self.dic_race['A'] = {}
                    self.dic_race['A']['num'] = 1
                    self.dic_race['A']['index'] = index_race
                    index_race += 1
                else:
                    self.dic_race['A']['num'] += 1
                if i not in self.dic_demographic:
                    self.dic_demographic[i] = {}
                    self.dic_demographic[i]['race'] = 'A'
            elif race[0] == 'B':
                if 'B' not in self.dic_race:
                    self.dic_race['B'] = {}
                    self.dic_race['B']['num'] = 1
                    self.dic_race['B']['index'] = index_race
                    index_race += 1
                else:
                    self.dic_race['B']['num'] += 1
                if i not in self.dic_demographic:
                    self.dic_demographic[i] = {}
                    self.dic_demographic[i]['race'] = 'B'
            elif race[0] == '<':
                race_ = race.split('>')[3].split('<')[0]
                if race_ not in self.dic_race:
                    self.dic_race[race_] = {}
                    self.dic_race[race_]['num'] = 1
                    self.dic_race[race_]['index'] = index_race
                    index_race += 1
                else:
                    self.dic_race[race_]['num'] += 1
                if i not in self.dic_demographic:
                    self.dic_demographic[i] = {}
                    self.dic_demographic[i]['race'] = race_
            elif race[0] == 'I' or race[0] == 'P':
                if 'U' not in self.dic_race:
                    self.dic_race['U'] = {}
                    self.dic_race['U']['num'] = 1
                    self.dic_race['U']['index'] = index_race
                    index_race += 1
                else:
                    self.dic_race['U']['num'] += 1
                if i not in self.dic_demographic:
                    self.dic_demographic[i] = {}
                    self.dic_demographic[i]['race'] = 'U'
            else:
                if race not in self.dic_race:
                    self.dic_race[race] = {}
                    self.dic_race[race]['num'] = 1
                    self.dic_race[race]['index'] = index_race
                    index_race += 1
                else:
                    self.dic_race[race]['num'] += 1
                if i not in self.dic_demographic:
                    self.dic_demographic[i] = {}
                    self.dic_demographic[i]['race'] = race
            if 'Age' not in self.dic_race:
                self.dic_race['Age'] = {}
                self.dic_race['Age']['index'] = index_race
                index_race += 1
            self.dic_demographic[i]['Age'] = age
            # index_race += 1
            if 'M' not in self.dic_race:
                self.dic_race['M'] = {}
                self.dic_race['M']['index'] = index_race
                index_race += 1
            if 'F' not in self.dic_race:
                self.dic_race['F'] = {}
                self.dic_race['F']['index'] = index_race
                index_race += 1
            self.dic_demographic[i]['gender'] = gender


        index = 0
        for i in self.dic_patient.keys():
            print(index)
            index += 1
            #in_icu_date = self.reg_ar
            self.single_patient_vital = np.where(self.vital_sign_ar[:, 0] == i)[0]
            in_time_value = self.dic_patient[i]['Admit_time_values'][0]
            self.single_patient_lab = np.where(self.labtest_ar[:, 0] == i)[0]
            total_value_lab = 0

            for k in self.single_patient_lab:
                obv_id = self.labtest_ar[k][2]
                patient_lab_mrn = self.labtest_ar[k][0]
                value = self.labtest_ar[k][3]
                self.check_data_lab = self.labtest_ar[k][4]
                date_year_value_lab = float(str(self.labtest_ar[k][4])[0:4]) * 365
                date_day_value_lab = float(str(self.check_data_lab)[4:6]) * 30 + float(str(self.check_data_lab)[6:8])
                date_value_lab = (date_year_value_lab + date_day_value_lab) * 24 * 60
                date_time_value_lab = float(str(self.check_data_lab)[8:10]) * 60 + float(
                    str(self.check_data_lab)[10:12])
                self.total_time_value_lab = date_value_lab + date_time_value_lab
                self.dic_patient[i].setdefault('lab_time_check', []).append(self.check_data_lab)
                if obv_id in self.dic_lab_category.keys():
                    category = self.dic_lab_category[obv_id]
                    self.prior_time = np.int(np.floor(np.float((self.total_time_value_lab - in_time_value) / 60)))
                    if self.prior_time < 0:
                        continue
                    try:
                        value = float(value)
                    except:
                        continue
                    if not value == value:
                        continue
                    if i not in self.dic_lab[category]:
                        # self.dic_lab[category]['patient_values'][i]={}
                        self.dic_lab[category].setdefault('lab_value_patient', []).append(value)
                    else:
                        self.dic_lab[category].setdefault('lab_value_patient', []).append(value)
                    if self.prior_time not in self.dic_patient[i]['prior_time_lab']:
                        self.dic_patient[i]['prior_time_lab'][self.prior_time] = {}
                        self.dic_patient[i]['prior_time_lab'][self.prior_time].setdefault(category, []).append(value)
                    else:
                        self.dic_patient[i]['prior_time_lab'][self.prior_time].setdefault(category, []).append(value)
            # if not self.dic_lab[category]['patient_values'][i] == {}:
            #   mean_value_lab_single = np.mean(self.dic_lab[category]['patient_values'][i]['lab_value_patient'])
            #  self.dic_lab[category]['patient_values'][i]['lab_mean_value']=mean_value_lab_single

            # print(index)
            # index += 1
            for j in self.single_patient_vital:
                obv_id = self.vital_sign_ar[j][2]
                if obv_id in self.crucial_vital:
                    self.check_data = self.vital_sign_ar[j][4]
                    self.dic_patient[i].setdefault('time_capture', []).append(self.check_data)
                    date_year_value = float(str(self.vital_sign_ar[j][4])[0:4]) * 365
                    date_day_value = float(str(self.check_data)[4:6]) * 30 + float(str(self.check_data)[6:8])
                    date_value = (date_year_value + date_day_value) * 24 * 60
                    date_time_value = float(str(self.check_data)[8:10]) * 60 + float(str(self.check_data)[10:12])
                    total_time_value = date_value + date_time_value
                    self.prior_time = np.int(np.floor(np.float((total_time_value - in_time_value) / 60)))
                    if self.prior_time < 0:
                        continue
                    if obv_id == 'CAC - BLOOD PRESSURE':
                        self.check_obv = obv_id
                        self.check_ar = self.vital_sign_ar[j]
                        self.check_value_presure = self.vital_sign_ar[j][3]
                        try:
                            value = self.vital_sign_ar[j][3].split('/')
                        except:
                            continue
                        if self.check_value_presure == '""':
                            continue
                        if self.prior_time not in self.dic_patient[i]['prior_time_vital']:
                            self.dic_patient[i]['prior_time_vital'][self.prior_time] = {}
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault('high', []).append(
                                value[0])
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault('low', []).append(
                                value[1])
                        else:
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault('high', []).append(
                                value[0])
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault('low', []).append(
                                value[1])
                        self.dic_vital['high'].setdefault('value', []).append(value[0])
                        self.dic_vital['low'].setdefault('value', []).append(value[1])
                    else:
                        self.check_value = self.vital_sign_ar[j][3]
                        self.check_obv = obv_id
                        self.check_ar = self.vital_sign_ar[j]
                        if self.check_value == '""':
                            continue
                        value = float(self.vital_sign_ar[j][3])
                        if np.isnan(value):
                            continue
                        if self.prior_time not in self.dic_patient[i]['prior_time_vital']:
                            self.dic_patient[i]['prior_time_vital'][self.prior_time] = {}
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault(obv_id, []).append(
                                value)
                        else:
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault(obv_id, []).append(
                                value)
                        self.dic_vital[obv_id].setdefault('value', []).append(value)

    def gen_demo_csv(self, name_to_store):
        pick_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 19,
                    20, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 36, 37, 38, 41, 43,
                    45, 46, 47, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 66,
                    67, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 83]
        pick_num = np.array(pick_num)
        self.feature_mean = []
        feature = list(np.array(list(self.dic_vital.keys()) + list(self.dic_lab.keys()))[pick_num])
        self.feature_copy = []
        self.feature_iqr = []
        feature_csv = feature #+ feature + feature + feature
        for i in feature:
            if i in self.dic_vital.keys():
                #self.feature_mean.append(self.dic_vital[i]['mean_value'])
                #mean = self.dic_vital[i]['mean_value']
                self.feature_copy.append(i)
                #std = np.float(self.dic_vital[i]['std'])
                values = [np.float(i) for i in self.dic_vital[i]['value'] if np.float(i)<mean+std]
                percent_75 = np.percentile(values,75)
                values_correction = [i for i in values if i < percent_75]
                mean_value = np.mean(values_correction)
                self.feature_mean.append(mean_value)
                irq_value = iqr(values_correction)
                self.feature_iqr.append(irq_value)
            if i in self.dic_lab.keys():
                #self.feature_mean.append(self.dic_lab[i]['mean_value'])
                #mean = self.dic_lab[i]['mean_value']
                self.feature_copy.append(i)
                #std = np.float(self.dic_labl[i]['std'])
                values = [np.float(i) for i in self.dic_lab[i]['lab_value_patient'] if np.float(i)<mean+std]
                percent_75 = np.percentile(values, 75)
                values_correction = [i for i in values if i < percent_75]
                mean_value = np.mean(values_correction)
                self.feature_mean.append(mean_value)
                irq_value = iqr(values_correction)
                self.feature_iqr.append(irq_value)

        #time_seq = list(np.ones(63)) + list(2 * np.ones(63)) + list(3 * np.ones(63)) + list(4 * np.ones(63))
        #time_step1 = self.ave_data_scores_total[0, :][pick_num]
        #time_step2 = self.ave_data_scores_total[1, :][pick_num]
        #time_step3 = self.ave_data_scores_total[2, :][pick_num]
        #time_step4 = self.ave_data_scores_total[3, :][pick_num]
        #variable_scores = list(time_step1) + list(time_step2) + list(time_step3) + list(time_step4)
        df = pd.DataFrame(
            {"Demographic Features": self.feature_copy, "mean_value": self.feature_mean,"irq":self.feature_iqr})
        df.to_csv(name_to_store, index=False)

    def gen_race_csv(self,name_to_store):
        self.dic_race_sec = {}
        self.dic_gen_sec = {}
        self.age_list = []
        for i in self.total_data_mortality:
            race = self.dic_demographic[i]['race']
            gender = self.dic_demographic[i]['gender']
            if race not in self.dic_race_sec.keys():
                self.dic_race_sec[race] = {}
                self.dic_race_sec[race] = 1
            else:
                self.dic_race_sec[race] += 1

            if gender not in self.dic_gen_sec.keys():
                self.dic_gen_sec[gender] = {}
                self.dic_gen_sec[gender] = 1
            else:
                self.dic_gen_sec[gender] += 1
            age = self.dic_demographic[i]['Age']
            self.age_list.append(age)

    def get_lab(self):
        self.dic_lab_count = {}
        for i in self.dic_lab.keys():
            self.dic_lab_count[i]=0
        for i in self.dic_patient.keys():
            m = []
            for j in self.dic_patient[i]['prior_time_lab'].keys():
                m += list(self.dic_patient[i]['prior_time_lab'][j].keys())
            for k in self.dic_lab.keys():
                if k in m:
                    self.dic_lab_count[k] += 1







        #df = pd.DataFrame(
           # {"Demographic Features": self.feature_copy, "mean_value": self.feature_mean, "irq": self.feature_iqr})
        #df.to_csv(name_to_store, index=False)

if __name__ == "__main__":
    kg = Kg_construct_ehr()
    kg.read_csv()
    #kg.create_kg_dic()

    with open('/datadrive/tingyi_wanyan/dic_patient_whole.json', 'r') as fp:
        kg.dic_patient = json.load(fp)
    with open('/datadrive/tingyi_wanyan/dic_vital_whole.json','r') as tp:
        kg.dic_vital = json.load(tp)
    with open('/datadrive/tingyi_wanyan/dic_demographic_whole.json','r') as fp_:
        kg.dic_demographic = json.load(fp_)
    with open('/datadrive/tingyi_wanyan/dic_race_whole.json','r') as tp_:
        kg.dic_race = json.load(tp_)
    with open('/datadrive/tingyi_wanyan/dic_lab_whole.json','r') as lab:
        kg.dic_lab = json.load(lab)


    for i in kg.dic_lab.keys():
        mean_lab = np.mean(kg.dic_lab[i]['lab_value_patient'])
        std_lab = np.mean(kg.dic_lab[i]['lab_value_patient'])
        kg.dic_lab[i]['mean_value'] = mean_lab
        kg.dic_lab[i]['std'] = std_lab

    for i in kg.dic_vital.keys():
        values = [np.float(j) for j in kg.dic_vital[i]['value']]
        mean = np.mean(values)
        std = np.std(values)
        kg.dic_vital[i]['mean_value'] = mean
        kg.dic_vital[i]['std'] = std



    kg.dic_death = {}
    kg.dic_intubation = {}
    kg.dic_in_icu = {}

    kg.total_intubation_time = []
    kg.total_in_icu_time = []
    kg.total_death_time = []

    kg.total_data_mortality = []
    kg.un_correct_mortality = []
    kg.total_data_intubation = []
    kg.un_correct_intubation = []
    kg.total_data_icu = []
    kg.un_correct_icu = []

    for i in kg.dic_demographic.keys():
        if kg.dic_patient[i]['death_flag'] == 0:
            kg.total_data_mortality.append(i)
            kg.dic_death.setdefault(0, []).append(i)
        if kg.dic_patient[i]['death_flag'] == 1:
            if kg.dic_patient[i]['death_hour'] > 0:
                kg.total_data_mortality.append(i)
                kg.dic_death.setdefault(1, []).append(i)
                kg.total_death_time.append(kg.dic_patient[i]['death_hour'])
            else:
                kg.un_correct_mortality.append(i)
        if kg.dic_patient[i]['intubation_label'] == 0:
            kg.total_data_intubation.append(i)
            kg.dic_intubation.setdefault(0, []).append(i)
        if kg.dic_patient[i]['intubation_label'] == 1:
            if kg.dic_patient[i]['intubation_hour'] > 0:
                kg.dic_intubation.setdefault(1, []).append(i)
                kg.total_data_intubation.append(i)
                kg.total_intubation_time.append(kg.dic_patient[i]['intubation_hour'])
            else:
                kg.un_correct_intubation.append(i)
        if kg.dic_patient[i]['icu_label'] == 0:
            kg.total_data_icu.append(i)
            kg.dic_in_icu.setdefault(0, []).append(i)
        if kg.dic_patient[i]['icu_label'] == 1:
            if kg.dic_patient[i]['in_icu_hour'] > 0:
                kg.total_data_icu.append(i)
                kg.dic_in_icu.setdefault(1, []).append(i)
                kg.total_in_icu_time.append(kg.dic_patient[i]['in_icu_hour'])
            else:
                kg.un_correct_icu.append(i)

    for i in kg.dic_patient.keys():
        if len(kg.dic_patient[i]['Admit_time_values']) > 1:
            first = kg.dic_patient[i]['Admit_time_values'][0]
            sec = kg.dic_patient[i]['Admit_time_values'][1]
            upper_bound = np.floor((sec - first) / 60)
            keys = kg.dic_patient[i]['prior_time_vital'].keys()
            pick_time = np.median([float(i) for i in keys if float(i) < upper_bound])
            kg.dic_patient[i]['pick_time'] = pick_time
            if not pick_time == pick_time:
                kg.dic_patient[i]['pick_time'] = 0
        else:
            keys = kg.dic_patient[i]['prior_time_vital'].keys()
            pick_time = np.median([float(i) for i in keys])
            kg.dic_patient[i]['pick_time'] = pick_time
            if not pick_time == pick_time:
                kg.dic_patient[i]['pick_time'] = 0


    BIPD_pick = np.where(kg.reg_ar[:,37]=="BIPD")
    In_patient_pick = np.where(kg.reg_ar[:,52]=="I")[0]
    In_patient_mrn = list(kg.reg_ar[:,45][In_patient_pick])
    BIPD_mrn_pick = list(kg.reg_ar[:,45][BIPD_pick])
    BIPD_intersect = np.intersect1d(BIPD_mrn_pick,list(kg.dic_patient.keys()))
    Mortality_intersect = np.intersect1d(kg.total_data_mortality,In_patient_mrn)
    Data_mortality = [i for i in Mortality_intersect if i not in BIPD_intersect]
    kg.total_data_mortality = Data_mortality
    intubation_intersect = np.intersect1d(kg.total_data_intubation,In_patient_mrn)
    Data_intubation = [i for i in intubation_intersect if i not in BIPD_intersect]
    kg.total_data_intubation = Data_intubation
    icu_intersect = np.intersect1d(kg.total_data_icu,In_patient_mrn)
    Data_icu = [i for i in icu_intersect if i not in BIPD_intersect]
    kg.total_data_icu = Data_icu


    kg.mean_death_time = np.mean(kg.total_death_time)
    kg.std_death_time = np.std(kg.total_death_time)
    kg.mean_intubate_time = np.mean(kg.total_intubation_time)
    kg.std_intubate_time = np.std(kg.total_intubation_time)
    kg.mean_icu_time = np.mean(kg.total_in_icu_time)
    kg.std_icu_time = np.std(kg.total_in_icu_time)

    age_total = []
    for i in kg.dic_demographic.keys():
        age = kg.dic_demographic[i]['Age']
        if age == 0:
            continue
        else:
            age_total.append(age)
    kg.age_mean = np.mean(age_total)
    kg.age_std = np.std(age_total)

    death_data = []
    for i in kg.dic_patient.keys():
        if kg.dic_patient[i]['death_flag']==1:
            if i in kg.total_data_mortality:
                death_data.append(i)

    intubate_data = []
    for i in kg.dic_patient.keys():
        if kg.dic_patient[i]['intubation_label']==1:
            if i in kg.total_data_intubation:
                intubate_data.append(i)

    icu_data = []
    for i in kg.dic_patient.keys():
        if kg.dic_patient[i]['icu_label'] == 1:
            if i in kg.total_data_icu:
                icu_data.append(i)

    #random_pick_death = random.sample(death_data,1200)

    random_pick_death = list(np.array(death_data)[0:1200])
    reduced_data = [i for i in kg.total_data_mortality if i not in random_pick_death]
    #kg.total_data_mortality = reduced_data
    kg.total_non_death_data = [i for i in kg.total_data_mortality if i not in death_data]


    random_pick_intubate = random.sample(intubate_data, 200)
    reduced_data_intubate = [i for i in kg.total_data_intubation if i not in random_pick_intubate]
    #kg.total_data_intubation = reduced_data_intubate

    random_pick_icu = random.sample(icu_data, 350)
    reduced_data_icu = [i for i in kg.total_data_icu if i not in random_pick_icu]
    kg.total_data_icu = reduced_data_icu

    """
    Demographic table stat
    """
    process_data = kg_process_data(kg)
    process_data.separate_train_test()
    #LSTM_ = LSTM_model(kg, process_data)
    #pretrain = pretrain_dhgm(kg,process_data)
    # LSTM_.config_model()
    # LSTM_.train()
    dhgm = dynamic_hgm(kg, process_data)
    contrastive = knn_cl(kg,process_data)
    #multi = Multiplex_network(kg)






