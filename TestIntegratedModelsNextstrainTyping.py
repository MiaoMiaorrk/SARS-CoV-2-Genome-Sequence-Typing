#The framework of this script refers to PangoLEARN: https://github.com/cov-lineages/pangoLEARN/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, make_scorer
from datetime import datetime
import joblib
import sys
import os
import time
from sklearn.model_selection import cross_val_score
from Bio import SeqIO
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

import numpy as np
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import csv
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

import gc

import seaborn as sns
from sklearn.metrics import confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

cout_flag = 0
debug_model = [1,1,1,1,1,1,1] #1-MLP 2-Catboost 3-RF 4-SVM 5-LogR 6-Ada 7-Decision

dataModetest = 1
if dataModetest == 0:
	print('~~~~~~~~~~~f3~~~~~~~~~~~~~one hot mode ->  n*5~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
else:
	print('~~~~~~~~~~~f2~~~~~~~~~~~~~nucleotide site mutation-based mode ->  n*1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

#1-MLP 2-Catboost 3-RF 4-SVM 5-LogR 6-Ada 7-Decision
weight_add = [0.05,0.25,0.25,0.1,0.15,0.20,0.00]
svm_predict_proba_valid = 0  #SVM_NSAll_Revised_Prob 1
print('_____________________weight_add: ',weight_add)

testing_percentage = 1.0
print('testing_percentage: ',testing_percentage)

lineage_file = './data/metadata.tsv'
sequence_file = './data/sequences.fasta'

print(lineage_file)
print(sequence_file)

loadTrainedModelFlag = 1
LoadOfflineIndiciesToKeepFile = 1
OffIndiciesFilePath = "./indiciesToKeep_464.txt"    #indiciesToKeep; indiciesToKeepS_N  indiciesToKeepNextstrain

hash_mode = 0  # default:0

if loadTrainedModelFlag == 1:
	if dataModetest == 0:
		model_file_1 = './models/ModelNextstrainMLP_f3.joblib'
	else:
		model_file_1 = './models/ModelNextstrainMLP_f2.joblib'
	base_loaded_model_1 = joblib.load(model_file_1)
	print('model_file_1 has been loaded: ',model_file_1)

	class_label = base_loaded_model_1.classes_
	for i in range(len(class_label)):
		class_label[i] = class_label[i][0:3]

	if dataModetest == 0:
		model_file_2 = './models/ModelNextstrainCatBoost_f3.joblib'
	else:
		model_file_2 = './models/ModelNextstrainCatBoost_f2.joblib'
	base_loaded_model_2 = joblib.load(model_file_2)
	print('model_file_2 has been loaded: ', model_file_2)

	if dataModetest == 0:
		model_file_3 = './models/ModelNextstrainRF_f3.joblib'
	else:
		model_file_3 = './models/ModelNextstrainRF_f2.joblib'
	base_loaded_model_3 = joblib.load(model_file_3)
	print('model_file_3 has been loaded: ', model_file_3)

	if dataModetest ==0:
		model_file_4 = './models/ModelNextstrainSVM_f3.joblib'   #prob = false -> SVM_NSAll_Revised; prob = true -> SVM_NSAll_Revised_Prob
	else:
		model_file_4 = './models/ModelNextstrainSVM_f2.joblib'
	base_loaded_model_4 = joblib.load(model_file_4)
	class_svm = base_loaded_model_4.classes_
	print('model_file_4 has been loaded: ', model_file_4)

	if dataModetest == 0:
		model_file_5 = './models/ModelNextstrainLR_f3.joblib'
	else:
		model_file_5 = './models/ModelNextstrainLR_f2.joblib'
	base_loaded_model_5 = joblib.load(model_file_5)
	print('model_file_5 has been loaded: ', model_file_5)

	if dataModetest == 0:
		model_file_6 = './models/ModelNextstrainAdaBoost_f3.joblib'
	else:
		model_file_6 = './models/ModelNextstrainAdaBoost_f2.joblib'
	base_loaded_model_6 = joblib.load(model_file_6)
	print('model_file_6 has been loaded: ', model_file_6)

	if dataModetest == 0:
		model_file_7 = './models/ModelNextstrainDT_f3.joblib'
	else:
		model_file_7 = './models/ModelNextstrainDT_f2.joblib'
	base_loaded_model_7 = joblib.load(model_file_7)
	print('model_file_7 has been loaded: ', model_file_7)

saveMissTypedFasta = 0

if saveMissTypedFasta == 1:
	inputFastaOrig = SeqIO.parse(sequence_file, "fasta")

referenceFile = './data/reference.fasta'

newSeqId = []

idToPangoType = dict()
idToGisaidType = dict()
dataList = []
indiciesToKeep = dict()
referenceId = "Wuhan/WH01/2019"
referenceSeq = ""

idToLineage = dict()
idToSeq = dict()

mustKeepIds = []
mustKeepLineages = []

def findReferenceSeq():
	with open(referenceFile) as f:
		currentSeq = ""

		for line in f:
			if ">" not in line:
				currentSeq = currentSeq + line.strip()

	f.close()
	return currentSeq


def getDataLine(seqId, seq):
	dataLine = []
	dataLine.append(seqId)
	dataLine.append(str(seq))
	return dataLine


def readInAndFormatData():
	# add the data line for the reference seq
	idToLineage[referenceId] = "A"
	dataList.append(getDataLine(referenceId, referenceSeq))

	tsvRes = []
	clades = []
	with open(lineage_file,'r') as f:
		next(f)
		for line in f:
			line_res = []
			cladeTypeIndex = 17
			line = line.strip('\n').split('\t')
			idtmp = line[0]
			line_lenth = len(line)

			#17:nextstrain types   18: pangolin   19:gisaid types
			idToLineage[line[0]] = line[cladeTypeIndex]    #17: nextstrain types   ;   19  gisaid types
			if cladeTypeIndex == 17:
				idToPangoType[line[0]] = line[18]
				idToGisaidType[line[0]] = line[19]

			clades.append(line[cladeTypeIndex])

	cladesdict = {}
	for key in clades:
		cladesdict[key] = cladesdict.get(key, 0) + 1
	print('\n'+'****************seq clades distribution******************')
	print(cladesdict)
	print('***********************************************************'+'\n')
	f.close()

	seq_dict = {rec.id : rec.seq for rec in SeqIO.parse(sequence_file, "fasta")}

	dataListSize = 0
	for key in seq_dict.keys():
		if key in idToLineage:
			dataList.append(getDataLine(key, seq_dict[key]))
			dataListSize += 1
			if dataListSize%10000==0:
				print('dataListSize: ',dataListSize)
		else:
			print("unable to find the lineage classification for: " + key)
	print('del seq_dict')
	del seq_dict
	gc.collect()

def removeOtherIndices(indiciesToKeep):

	finalList = []

	indicies = list(indiciesToKeep.keys())
	indicies.sort()

	while len(dataList) > 0:

		line = dataList.pop(0)
		seqId = line.pop(0)

		line = line[0]
		finalLine = []

		for index in indicies:
			if index == 0:
				finalLine.append(seqId)
			else:
				finalLine.append(line[index])
		finalList.append(finalLine)

	return finalList

def blackListing():
	newList = []

	print("keeping indicies:")

	for line in dataList:
		seqIdtmp = line[0]
		line[0] = idToLineage[line[0]]
		if line[0] == 'A':
			print('*******************Warning: remove the reference type!**********************')
		else:
			newList.append(line)
			newSeqId.append(seqIdtmp)

	return newList

referenceSeq = findReferenceSeq()

readInAndFormatData()

with open(OffIndiciesFilePath, "r") as f:
	data1 = f.readlines()

	indiciesToKeep[0] = True

	for i in data1:
		indiciesToKeep[int(i)] = True

dataList = removeOtherIndices(indiciesToKeep)

dataList = blackListing()

if dataModetest == 1:
	seq_num = len(dataList)
	seq_len = len(dataList[0]) -1

	refseq = dataList[0][1:]
	for index in range(1,seq_num):
		queryline = dataList[index][1:]
		for queryIndex in range(len(queryline)):
			query = queryline[queryIndex]
			ref   = refseq[queryIndex]
			if query == ref:
				dataList[index][queryIndex+1] = 0
			else:
				dataList[index][queryIndex+1] = 1

	for i in range(len(queryline)):
		dataList[0][i + 1] = 0

headers = list(indiciesToKeep.keys())
headers[0] = "lineage"

pima = pd.DataFrame(dataList, columns=headers)

if dataModetest == 0:
	categories = ['A', 'C', 'G', 'T', '-']
	dummyHeaders = headers[1:]

	for i in categories:
		line = [i] * len(dataList[0])
		pima.loc[len(pima)] = line

	pima = pd.get_dummies(pima, columns=dummyHeaders)

	pima.drop(pima.tail(len(categories)).index, inplace=True)

feature_cols = list(pima)

h = feature_cols.pop(0)
X = pima[feature_cols]
y = pima[h]

print('del dataList')
del dataList
gc.collect()

if loadTrainedModelFlag == 0:
	X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=testing_percentage,stratify=y)
else:
	if testing_percentage < 1:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing_percentage, stratify=y)
	else:
		X_test = X
		y_test = y
	test_data_size = len(y_test)
	print('test_data_size/total_data_size ',test_data_size,'/',len(y))


print('___________________load difference matrix_____________________')

file = open('./DM/diffM')
diffmatrix1 = file.read()
diffmatrix2 = diffmatrix1.split('\n')
diffmatrix2 = diffmatrix2[0:25]
for i in range(len(diffmatrix2)):
	diffmatrix2[i] = diffmatrix2[i].split()
len(diffmatrix2[0])
DM = np.array(diffmatrix2)  #25*25 in order 21M 21K 21L ...  19B 19A


file = open('./DM/clade_size')
clade_size = file.read()
clade_size = clade_size.split('\n')
clade_size = clade_size[0:25]
cladeSize = np.array(clade_size)


file = open('./DM/label')
label = file.read()
label = label.split('\n')
label = label[0:25]
label2ID = dict()
for i in range(len(label)):
	label_tmp = label[i]
	label2ID[label_tmp] = i

label2 = []
for i in range(len(label)):
	label2.append(label[len(label)-1-i])

shape = (25,25)
DM_diag = np.zeros(shape)
for i in range(len(label)):
	for j in range(len(label)):
		dij = float(DM[i][j])*float(cladeSize[i])/(float(cladeSize[i])+float(cladeSize[j])) \
			  + float(DM[j][i])*float(cladeSize[j])/(float(cladeSize[i])+float(cladeSize[j]))
		DM_diag[i][j] = dij
		if i==j:
			DM_diag[i][j] = 0
print('___________________load difference matrix end_____________________')

if loadTrainedModelFlag == 1:
	if saveMissTypedFasta == 1:
		fastaSave = open('./data/missTypedFasta.fasta', 'w', newline='')

	if debug_model[0] == 1:
		print('___________________________Testing trained model 1____MLP_______________________')
		y_pred_1 = base_loaded_model_1.predict(X_test)
		y_pred_list_1 = y_pred_1.tolist()
		predictions_1 = base_loaded_model_1.predict_proba(X_test)   #test
		print(metrics.classification_report(y_test, y_pred_1, digits=5))
		p = precision_score(y_test, y_pred_1, average='weighted')  #
		r = recall_score(y_test, y_pred_1, average='weighted')
		f1 = f1_score(y_test, y_pred_1, average='weighted')
		print('precision: ', p)
		print('recall: ', r)
		print('f1_score: ', f1)

		# compute AD
		weighted_error_score = 0
		err_num = 0
		test_y = y_test.to_list()
		y_true = []
		y_pred = []
		for i in range(len(test_y)):
			Predict = y_pred_1[i]
			GT = test_y[i]
			Predict = Predict[0:3]
			GT = GT[0:3]
			y_true.append(GT)
			y_pred.append(Predict)
			if GT != Predict:
				if cout_flag == 1:
					print('_______________', GT, Predict)
				if GT in label2ID and Predict in label2ID:
					id_i = label2ID[GT]
					id_j = label2ID[Predict]
					diff_w = DM_diag[id_i][id_j]
					weighted_error_score += diff_w
					err_num += 1
		if err_num > 0:
			weighted_error_score = weighted_error_score / err_num
			print('weighted_error_score: ', weighted_error_score)

	if debug_model[1] == 1:
		print('___________________________Testing trained model 2_____Catboost______________________')
		y_pred_2 = base_loaded_model_2.predict(X_test)
		y_pred_list_2 = y_pred_2.tolist()
		predictions_2 = base_loaded_model_2.predict_proba(X_test)  # test
		print(metrics.classification_report(y_test, y_pred_2, digits=5))
		p = precision_score(y_test, y_pred_2, average='weighted')  #
		r = recall_score(y_test, y_pred_2, average='weighted')
		f1 = f1_score(y_test, y_pred_2, average='weighted')
		print('precision: ', p)
		print('recall: ', r)
		print('f1_score: ', f1)

		# compute AD
		weighted_error_score = 0
		err_num = 0
		test_y = y_test.to_list()
		for i in range(len(test_y)):
			Predict = y_pred_2[i]
			GT = test_y[i]
			Predict = Predict[0][0:3]
			GT = GT[0:3]
			if GT != Predict:
				if cout_flag == 1:
					print('_______________', GT, Predict)
				if GT in label2ID and Predict in label2ID:
					id_i = label2ID[GT]
					id_j = label2ID[Predict]
					diff_w = DM_diag[id_i][id_j]
					weighted_error_score += diff_w
					err_num += 1
		if err_num > 0:
			weighted_error_score = weighted_error_score / err_num
			print('weighted_error_score: ', weighted_error_score)

	if debug_model[2] == 1:
		print('___________________________Testing trained model 3________RF___________________')
		y_pred_3 = base_loaded_model_3.predict(X_test)
		y_pred_list_3 = y_pred_3.tolist()
		predictions_3 = base_loaded_model_3.predict_proba(X_test)  # test
		print(metrics.classification_report(y_test, y_pred_3, digits=5))
		p = precision_score(y_test, y_pred_3, average='weighted')  #
		r = recall_score(y_test, y_pred_3, average='weighted')
		f1 = f1_score(y_test, y_pred_3, average='weighted')
		print('precision: ', p)
		print('recall: ', r)
		print('f1_score: ', f1)

		# compute AD
		weighted_error_score = 0
		err_num = 0
		test_y = y_test.to_list()
		for i in range(len(test_y)):
			Predict = y_pred_3[i]
			GT = test_y[i]
			Predict = Predict[0:3]
			GT = GT[0:3]
			if GT != Predict:
				if cout_flag == 1:
					print('_______________', GT, Predict)
				if GT in label2ID and Predict in label2ID:
					id_i = label2ID[GT]
					id_j = label2ID[Predict]
					diff_w = DM_diag[id_i][id_j]
					weighted_error_score += diff_w
					err_num += 1
		if err_num > 0:
			weighted_error_score = weighted_error_score / err_num
			print('weighted_error_score: ', weighted_error_score)

	if debug_model[3] == 1:
		print('___________________________Testing trained model 4________SVM___________________')
		y_pred_4 = base_loaded_model_4.predict(X_test)
		y_pred_list_4 = y_pred_4.tolist()
		if svm_predict_proba_valid == 1:
			predictions_4 = base_loaded_model_4.predict_proba(X_test)  # v2
		else:
			predictions_4 = [[0 for col in range(25)] for row in range(test_data_size)]   # v1
			for i in range(len(y_pred_list_4)):
				for j in range(len(base_loaded_model_4.classes_)):
					if y_pred_list_4[i] == base_loaded_model_4.classes_[j]:
						predictions_4[i][j] = 1
		print(metrics.classification_report(y_test, y_pred_4, digits=5))
		p = precision_score(y_test, y_pred_4, average='weighted')  #
		r = recall_score(y_test, y_pred_4, average='weighted')
		f1 = f1_score(y_test, y_pred_4, average='weighted')
		print('precision: ', p)
		print('recall: ', r)
		print('f1_score: ', f1)

		# compute AD
		weighted_error_score = 0
		err_num = 0
		test_y = y_test.to_list()
		for i in range(len(test_y)):
			Predict = y_pred_4[i]
			GT = test_y[i]
			Predict = Predict[0:3]
			GT = GT[0:3]
			if GT != Predict:
				if cout_flag == 1:
					print('_______________', GT, Predict)
				if GT in label2ID and Predict in label2ID:
					id_i = label2ID[GT]
					id_j = label2ID[Predict]
					diff_w = DM_diag[id_i][id_j]
					weighted_error_score += diff_w
					err_num += 1
		if err_num > 0:
			weighted_error_score = weighted_error_score / err_num
			print('weighted_error_score: ', weighted_error_score)

	if debug_model[4] == 1:
		print('___________________________Testing trained model 5______LogR_____________________')
		y_pred_5 = base_loaded_model_5.predict(X_test)
		y_pred_list_5 = y_pred_5.tolist()
		predictions_5 = base_loaded_model_5.predict_proba(X_test)  # test
		print(metrics.classification_report(y_test, y_pred_5, digits=5))
		p = precision_score(y_test, y_pred_5, average='weighted')  #
		r = recall_score(y_test, y_pred_5, average='weighted')
		f1 = f1_score(y_test, y_pred_5, average='weighted')
		print('precision: ', p)
		print('recall: ', r)
		print('f1_score: ', f1)

		# compute AD
		weighted_error_score = 0
		err_num = 0
		test_y = y_test.to_list()
		for i in range(len(test_y)):
			Predict = y_pred_5[i]
			GT = test_y[i]
			Predict = Predict[0:3]
			GT = GT[0:3]
			if GT != Predict:
				if cout_flag == 1:
					print('_______________', GT, Predict)
				if GT in label2ID and Predict in label2ID:
					id_i = label2ID[GT]
					id_j = label2ID[Predict]
					diff_w = DM_diag[id_i][id_j]
					weighted_error_score += diff_w
					err_num += 1
		if err_num > 0:
			weighted_error_score = weighted_error_score / err_num
			print('weighted_error_score: ', weighted_error_score)

	if debug_model[5] == 1:
		print('___________________________Testing trained model 6_________Ada__________________')
		y_pred_6 = base_loaded_model_6.predict(X_test)
		y_pred_list_6 = y_pred_6.tolist()
		predictions_6 = base_loaded_model_6.predict_proba(X_test)  # test
		print(metrics.classification_report(y_test, y_pred_6, digits=5))
		p = precision_score(y_test, y_pred_6, average='weighted')  #
		r = recall_score(y_test, y_pred_6, average='weighted')
		f1 = f1_score(y_test, y_pred_6, average='weighted')
		print('precision: ', p)
		print('recall: ', r)
		print('f1_score: ', f1)

		# compute AD
		weighted_error_score = 0
		err_num = 0
		test_y = y_test.to_list()
		for i in range(len(test_y)):
			Predict = y_pred_6[i]
			GT = test_y[i]
			Predict = Predict[0:3]
			GT = GT[0:3]
			if GT != Predict:
				if cout_flag == 1:
					print('_______________', GT, Predict)
				if GT in label2ID and Predict in label2ID:
					id_i = label2ID[GT]
					id_j = label2ID[Predict]
					diff_w = DM_diag[id_i][id_j]
					weighted_error_score += diff_w
					err_num += 1
		if err_num > 0:
			weighted_error_score = weighted_error_score / err_num
			print('weighted_error_score: ', weighted_error_score)

	if debug_model[6] == 1:
		print('___________________________Testing trained model 7_________Decision__________________')
		y_pred_7 = base_loaded_model_7.predict(X_test)
		y_pred_list_7 = y_pred_7.tolist()
		predictions_7 = base_loaded_model_7.predict_proba(X_test)  # test
		print(metrics.classification_report(y_test, y_pred_7, digits=5))
		p = precision_score(y_test, y_pred_7, average='weighted')  #
		r = recall_score(y_test, y_pred_7, average='weighted')
		f1 = f1_score(y_test, y_pred_7, average='weighted')
		print('precision: ', p)
		print('recall: ', r)
		print('f1_score: ', f1)

		# compute AD
		weighted_error_score = 0
		err_num = 0
		test_y = y_test.to_list()
		for i in range(len(test_y)):
			Predict = y_pred_7[i]
			GT = test_y[i]
			Predict = Predict[0:3]
			GT = GT[0:3]
			if GT != Predict:
				if cout_flag == 1:
					print('_______________', GT, Predict)
				if GT in label2ID and Predict in label2ID:
					id_i = label2ID[GT]
					id_j = label2ID[Predict]
					diff_w = DM_diag[id_i][id_j]
					weighted_error_score += diff_w
					err_num += 1
		if err_num > 0:
			weighted_error_score = weighted_error_score / err_num
			print('weighted_error_score: ', weighted_error_score)

	#test:
	error_num = 0
	low_score_num = 0
	weighted_error_score = 0
	y_true_list = []
	y_pred_list = []
	start_time = time.time()
	for i in range(test_data_size):
		if i%10000==0:
			print("testing step: ",i,' / ',test_data_size)
		x_in = X_test.iloc[i]
		y_real = y_test.iloc[i]
		x_id = newSeqId[i]

		#maxscore
		maxScore = 0
		predictions = [0] * 25
		for i_score in range(predictions_1.shape[1]):
			predictions[i_score] = weight_add[0] * predictions_1[i][i_score] + weight_add[1] * predictions_2[i][i_score] + weight_add[2] * \
								   predictions_3[i][i_score] + weight_add[3] * predictions_4[i][i_score] + weight_add[4] * \
								   predictions_5[i][i_score] + weight_add[5] * predictions_6[i][i_score] + weight_add[6] * \
								   predictions_7[i][i_score]

			if predictions[i_score] > maxScore:
				maxScore = predictions[i_score]
				maxIndex = i_score
		y_predict = base_loaded_model_1.classes_[maxIndex]
		if(maxScore<0.8):
			low_score_num += 1
			# print(i,'  seqId  ',x_id,'  maxscore: ',maxScore)

		if hash_mode == 1:
			y_real_2 = y_real[0:3]
			y_predict_2 = y_predict[0][0:3]
		else:
			y_real_2 = y_real[0:3]
			y_predict_2 = y_predict[0:3]

		y_pred_list.append(y_predict_2)
		y_true_list.append(y_real_2)

		if y_predict_2 != y_real_2:
			error_num += 1

			if y_real_2 in label2ID and y_predict_2 in label2ID:
				id_i = label2ID[y_real_2]
				id_j = label2ID[y_predict_2]
				diff_w = DM_diag[id_i][id_j]
				weighted_error_score += diff_w
				if cout_flag == 1:
					print('>>>', error_num, ' ', i, ' ', x_id, ' ', y_real,' ', y_predict,' ',maxScore,  ' diff_weight',diff_w)
			else:
				print("Error!",y_real_2,y_predict_2)

			if saveMissTypedFasta == 1:
				for seqRead in inputFastaOrig:
					seqId = seqRead.id
					seqSeq = seqRead.seq
					if seqId == x_id:
						saveSeq = seqSeq
						break
				fastaSave.write(">" + seqId + "\n")
				fastaSave.write(str(saveSeq) + "\n")
				print('saving fasta id: ',seqId)

	if error_num > 0:
		weighted_error_score = weighted_error_score/error_num
		print('weighted_error_score: ',weighted_error_score)
	error_rate = error_num/test_data_size
	accuracy = (1-error_rate)*100.0
	print('Accuracy: ')
	print('%.5f'%accuracy,'%')
	print('faild/total: ',error_num,'/',test_data_size)
	print('low_score_num: ',low_score_num)

	print('_________________________________Summary________________________________________')
	print(metrics.classification_report(y_true_list, y_pred_list, digits=5))
	p = precision_score(y_true_list, y_pred_list, average='weighted')  #
	r = recall_score(y_true_list, y_pred_list, average='weighted')
	f1 = f1_score(y_true_list, y_pred_list, average='weighted')
	acc = accuracy_score(y_true_list, y_pred_list)
	print('acc: ', acc)
	print('precision: ', p)
	print('recall: ', r)
	print('f1_score: ', f1)

	conf_mat = confusion_matrix(y_true_list, y_pred_list)
	row_size = np.sum(conf_mat, axis=1)
	conf_mat2 = np.zeros((len(row_size), len(row_size)), dtype=np.float)
	for i in range(len(row_size)):  # row
		for j in range(len(conf_mat[0])):
			if row_size[i] > 0:
				conf_mat2[i][j] = conf_mat[i][j] / row_size[i]
			else:
				if i == j:
					conf_mat2[i][j] = 1
				else:
					conf_mat2[i][j] = 0
	df_cm = pd.DataFrame(conf_mat2, index=class_label, columns=class_label)

	plt.figure(figsize=(20, 15))
	plt.rcParams['figure.dpi'] = 400
	heatmap = sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='YlGnBu', annot_kws={'size': 11.5})
	heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12,
								 fontweight='bold')
	heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=30, ha='right', fontsize=12,
								 fontweight='bold')
	plt.ylabel("True Label", fontsize=14, fontweight='bold')
	plt.xlabel("Predict Label", fontsize=14, fontweight='bold')
	plt.savefig("confusion_matrix_ratio.png", dpi=400, orientation='landscape')
	# plt.show()

	plt.close()

	conf_mat = confusion_matrix(y_true_list, y_pred_list)
	df_cm = pd.DataFrame(conf_mat, index=class_label, columns=class_label)

	plt.figure(figsize=(20, 15))
	plt.rcParams['figure.dpi'] = 400
	heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu', annot_kws={'size': 11.5})
	heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12,
								 fontweight='bold')
	heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=30, ha='right', fontsize=12,
								 fontweight='bold')
	plt.ylabel("True Label", fontsize=14, fontweight='bold')
	plt.xlabel("Predict Label", fontsize=14, fontweight='bold')
	plt.savefig("confusion_matrix_number.png", dpi=400, orientation='landscape')

