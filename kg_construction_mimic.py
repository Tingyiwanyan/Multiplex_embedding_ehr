import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
import math
import time
import pandas as pd
#from kg_model_modify import hetero_model_modify
#from LSTM_model import LSTM_model
#from kg_process_data import kg_process_data
#from Shallow_nn_ehr import NN_model
#from evaluation import cal_auc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class Kg_construct_ehr():
    """
    construct knowledge graph out of EHR data
    """
    def __init__(self):
        file_path = '/home/tingyi/MIMIC'
        self.diagnosis = file_path + '/DIAGNOSES_ICD.csv'
        self.diagnosis_d = file_path + '/D_ICD_DIAGNOSES.csv'
        self.prescription = file_path + '/PRESCRIPTIONS.csv'
        self.charteve = file_path + '/CHARTEVENTS.csv'
        self.d_item = file_path + '/D_ITEMS.csv'
        self.noteevents = file_path + '/NOTEEVENTS.csv'
        self.proc_icd = file_path + '/PROCEDURES_ICD.csv'
        self.patient_info = file_path + '/PATIENTS.csv'
        self.read_diagnosis()
        self.read_charteve()
        self.read_diagnosis_d()
        self.read_prescription()
        self.read_ditem()
        self.threshold_freq = 60
        #self.read_proc_icd()

    def read_diagnosis(self):
        self.diag = pd.read_csv(self.diagnosis)
        self.diag_ar = np.array(self.diag)

    def read_diagnosis_d(self):
        self.diag_d = pd.read_csv(self.diagnosis_d)
        self.diag_d_ar = np.array(self.diag_d)

    def read_prescription(self):
        self.pres = pd.read_csv(self.prescription)

    def read_charteve(self):
        self.char = open(self.charteve)
        #self.char = pd.read_csv(self.charteve,chunksize=4000000)
        #self.char_ar = np.array(self.char.get_chunk())
        #self.num_char = self.char_ar.shape[0]

    def read_patient(self):
        self.patient = pd.read_csv(self.patient_info)

    def read_ditem(self):
        self.d_item = pd.read_csv(self.d_item)
        self.d_item_ar = np.array(self.d_item)

    def read_noteevent(self):
        self.note = pd.read_csv(self.noteevents,chunksize=1000)

    def read_proc_icd(self):
        self.proc_icd = pd.read_csv(self.proc_icd)

    def create_kg_dic(self):
        self.dic_patient = {}
        self.dic_diag = {}
        self.dic_item = {}
        self.dic_patient_addmission = {}
        index_item = 0
        index_diag = 0
        num_read = 0
        for line in self.char:
            if num_read == 0:
                num_read += 1
                continue
            if num_read > 5000000:
                break
            num_read += 1
            line = line.rstrip('\n')
            line = line.split(',')
            itemid = np.int(line[4])
            value = np.float(line[9])
            hadm_id = np.int(line[2])
            patient_id = np.int(line[1])
            date_time = line[5].split(' ')
            date = [np.int(i) for i in date_time[0].split('-')]
            date_value = date[0]*10000+date[1]*100+date[2]
            time = [np.int(i) for i in date_time[1].split(':')]
            time_value = time[0]*100 + time[1]
            date_time_value = date_value*10000+time_value
            if itemid not in self.dic_item:
                self.dic_item[itemid] = {}
                self.dic_item[itemid]['nodetype'] = 'item'
                self.dic_item[itemid].setdefault('neighbor_patient', []).append(hadm_id)
                self.dic_item[itemid]['item_index'] = index_item
                self.dic_item[itemid].setdefault('value', []).append(value)
                #self.dic_item[itemid]['mean_value'] = []
                index_item += 1
            else:
                self.dic_item[itemid].setdefault('value', []).append(value)
                if hadm_id not in self.dic_item[itemid]['neighbor_patient']:
                    self.dic_item[itemid].setdefault('neighbor_patient', []).append(hadm_id)

            if hadm_id not in self.dic_patient:
                self.dic_patient[hadm_id] = {}
                self.dic_patient[hadm_id]['itemid'] = {}
                self.dic_patient[hadm_id]['nodetype'] = 'patient'
                self.dic_patient[hadm_id]['next_admission'] = None
                self.dic_patient[hadm_id]['itemid'].setdefault(itemid, []).append(value)
                #self.dic_patient[patient_id]['neighbor_presc'] = {}
            else:
                self.dic_patient[hadm_id]['itemid'].setdefault(itemid,[]).append(value)

            if patient_id not in self.dic_patient_addmission:
                self.dic_patient_addmission[patient_id] = {}
                self.dic_patient_addmission[patient_id][hadm_id] = {}
                self.dic_patient_addmission[patient_id][hadm_id]['date_time'] = date_time
                self.dic_patient_addmission[patient_id][hadm_id]['date'] = date
                self.dic_patient_addmission[patient_id][hadm_id]['date_value'] = date_value
                self.dic_patient_addmission[patient_id][hadm_id]['time'] = time
                self.dic_patient_addmission[patient_id][hadm_id]['time_value'] = time_value
                self.dic_patient_addmission[patient_id][hadm_id]['date_time_value'] = date_time_value
                self.dic_patient_addmission[patient_id].setdefault('time_series',[]).append(hadm_id)
            else:
                if hadm_id not in self.dic_patient_addmission[patient_id]['time_series']:
                    self.dic_patient_addmission[patient_id][hadm_id] = {}
                    self.dic_patient_addmission[patient_id][hadm_id]['date_time'] = date_time
                    self.dic_patient_addmission[patient_id][hadm_id]['date'] = date
                    self.dic_patient_addmission[patient_id][hadm_id]['time'] = time
                    self.dic_patient_addmission[patient_id][hadm_id]['date_value'] = date_value
                    self.dic_patient_addmission[patient_id][hadm_id]['time_value'] = time_value
                    self.dic_patient_addmission[patient_id][hadm_id]['date_time_value'] = date_time_value
                    index = 0
                    flag = 0
                    for i in self.dic_patient_addmission[patient_id]['time_series']:
                        if date_time_value < self.dic_patient_addmission[patient_id][i]['date_time_value']:
                            self.dic_patient_addmission[patient_id]['time_series'].insert(index,hadm_id)
                            self.dic_patient[hadm_id]['next_admission'] = i
                            if not index == 0:
                                self.dic_patient[self.dic_patient_addmission[patient_id]['time_series'][index-1]]['next_admission'] = hadm_id
                            flag = 1
                            break
                        index += 1
                    if flag == 0:
                        self.dic_patient[self.dic_patient_addmission[patient_id]['time_series'][-1]]['next_admission'] = hadm_id
                        self.dic_patient_addmission[patient_id].setdefault('time_series', []).append(hadm_id)





            #self.dic_patient.setdefault('itemid',[]).append([])

        for i in range(self.diag_ar.shape[0]):
            hadm_id = self.diag_ar[i][2]
            diag_icd = self.diag_ar[i][4]
            if hadm_id in self.dic_patient:
                if diag_icd not in self.dic_diag:
                    self.dic_diag[diag_icd] = {}
                    self.dic_diag[diag_icd].setdefault('neighbor_patient', []).append(hadm_id)
                    self.dic_diag[diag_icd]['nodetype'] = 'diagnosis'
                    self.dic_diag[diag_icd]['diag_index'] = index_diag
                    self.dic_patient[hadm_id].setdefault('neighbor_diag', []).append(diag_icd)
                    index_diag += 1
                else:
                    self.dic_patient[hadm_id].setdefault('neighbor_diag',[]).append(diag_icd)
                    self.dic_diag[diag_icd].setdefault('neighbor_patient',[]).append(hadm_id)

        for i in self.dic_item.keys():
            self.dic_item[i]['mean_value'] = np.mean(self.dic_item[i]['value'])
            self.dic_item[i]['std'] = np.std(self.dic_item[i]['value'])

        for k in self.dic_diag.keys():
            num_k1 = k[0]
            num_k3 = k[0:3]
            if (num_k1 == 'E' or num_k1 == 'V'):
                self.dic_diag[k]['icd'] = 17
                self.dic_diag[k]['icd_type'] = 'E and V codes: external causes of injury and supplemental classification'
            elif int(num_k3) <= 139:
                self.dic_diag[k]['icd'] = 0
                self.dic_diag[k]['icd_type'] = '001–139: infectious and parasitic diseases'
            elif int(num_k3) <= 239:
                self.dic_diag[k]['icd'] = 1
                self.dic_diag[k]['icd_type'] = '140–239: neoplasms'
            elif int(num_k3) <= 279:
                self.dic_diag[k]['icd'] = 2
                self.dic_diag[k]['icd_type'] = '240–279: endocrine, nutritional and metabolic diseases, and immunity disorders'
            elif int(num_k3) <= 289:
                self.dic_diag[k]['icd'] = 3
                self.dic_diag[k]['icd_type'] = '280–289: diseases of the blood and blood-forming organs'
            elif int(num_k3) <= 319:
                self.dic_diag[k]['icd'] = 4
                self.dic_diag[k]['icd_type'] = '290–319: mental disorders'
            elif int(num_k3) <= 389:
                self.dic_diag[k]['icd'] = 5
                self.dic_diag[k]['icd_type'] = '320–389: diseases of the nervous system and sense organs'
            elif int(num_k3) <= 459:
                self.dic_diag[k]['icd'] = 6
                self.dic_diag[k]['icd_type'] = '390–459: diseases of the circulatory system'
            elif int(num_k3) <= 519:
                self.dic_diag[k]['icd'] = 7
                self.dic_diag[k]['icd_type'] = '460–519: diseases of the respiratory system'
            elif int(num_k3) <= 579:
                self.dic_diag[k]['icd'] = 8
                self.dic_diag[k]['icd_type'] = '520–579: diseases of the digestive system'
            elif int(num_k3) <= 629:
                self.dic_diag[k]['icd'] = 9
                self.dic_diag[k]['icd_type'] = '580–629: diseases of the genitourinary system'
            elif int(num_k3) <= 679:
                self.dic_diag[k]['icd'] = 10
                self.dic_diag[k]['icd_type'] = '630–679: complications of pregnancy, childbirth, and the puerperium'
            elif int(num_k3) <= 709:
                self.dic_diag[k]['icd'] = 11
                self.dic_diag[k]['icd_type'] = '680–709: diseases of the skin and subcutaneous tissue'
            elif int(num_k3) <= 739:
                self.dic_diag[k]['icd'] = 12
                self.dic_diag[k]['icd_type'] = '710–739: diseases of the musculoskeletal system and connective tissue'
            elif int(num_k3) <= 759:
                self.dic_diag[k]['icd'] = 13
                self.dic_diag[k]['icd_type'] = '740–759: congenital anomalies'
            elif int(num_k3) <= 779:
                self.dic_diag[k]['icd'] = 14
                self.dic_diag[k]['icd_type'] = '760–779: certain conditions originating in the perinatal period'
            elif int(num_k3) <= 799:
                self.dic_diag[k]['icd'] = 15
                self.dic_diag[k]['icd_type'] = '780–799: symptoms, signs, and ill-defined conditions'
            else:
                self.dic_diag[k]['icd'] = 16
                self.dic_diag[k]['icd_type'] = '800–999: injury and poisoning'





    def create_kg(self):
        self.g = nx.DiGraph()
        for i in range(self.num_char):
            patient_id = self.char_ar[i][1]/home/tingyi/ecgtoolkit-cs-git/ECGToolkit/libs/ECGConversion/MUSEXML/MUSEXML/MUSEXMLFormat.cs
            itemid = self.char_ar[i][4]
            value = self.char_ar[i][8]
            itemid_list = np.where(self.d_item_ar == itemid)
            diag_list = np.where(self.diag_ar[:,1] == patient_id)
            diag_icd9_list = self.diag_ar[:,4][diag_list]
            diag_d_list = [np.where(self.diag_d_ar[:,1] == diag_icd9_list[x])[0] for x in range(diag_icd9_list.shape[0])]
            """
            Add patient node
            """
            self.g.add_node(patient_id, item_id=itemid)
            self.g.add_node(patient_id, test_value=value)
            self.g.add_node(patient_id, node_type='patient')
            self.g.add_node(patient_id, itemid_list=itemid_list)
            self.g.add_node(itemid, node_type='ICD9')
            """
            Add diagnosis ICD9 node
            """
            self.g.add_edge(patient_id, itemid, type='')

    def diag_freq(self):
        self.frequent_diag = []
        index_freq = 0
        for i in self.dic_diag.keys():
            num_patient = len(self.dic_diag[i]['neighbor_patient'])
            if num_patient > self.threshold_freq:
                self.dic_diag[i]['index_freq'] = index_freq
                self.frequent_diag.append(i)
                index_freq += 1

if __name__ == "__main__":
    kg = Kg_construct_ehr()
    kg.create_kg_dic()
    kg.diag_freq()
    process_data = kg_process_data(kg)
    test_accur = []
    #process_data.seperate_train_test()
    print("finished loading")

    #for i in range(3):
    """
    process_data.cross_validation_seperate(0)
    process_data.filter_patient_diag()
    hetro_model = hetero_model_modify(kg,process_data)
    #LSTM_model = LSTM_model(kg,hetro_model,process_data)
    hetro_model.config_model()
    print("in get train")
    hetro_model.get_train_graph()
    hetro_model.train()
    """