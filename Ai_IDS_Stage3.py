import csv
from datetime import datetime
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
import scikitplot as skP

# from sklearn.metrics import hamming_loss

startTime = datetime.now()

# clear previous content before writing
txtClear = open("output.txt", "w").close()

DaFe = pd.read_csv('kddcup_features.txt')
# print(DaFe.columns.shape)

DaSt1 = pd.read_csv('kddcup.data_10_percent_corrected.csv', names=DaFe.columns, header=None)
# DaTest = pd.read_csv('kddcup.newtestdata_10_percent_unlabeled', names=DaFe.columns, header=None)
# None-Header removes first row in dataset as header

DaSt1.dropna()
DaSt1.drop_duplicates(keep='first', inplace=True)

label = DaSt1['intrusion_type']
labels = label.values.ravel()
# print(labels)

DaSt = DaSt1.drop(['intrusion_type'], axis=1)
LablHtEncding = preprocessing.LabelEncoder()
DaSt = DaSt.apply(LablHtEncding.fit_transform)
# labels = labelz.apply(LablHtEncding.fit_transform())

# print(labels)
# LablHtEncding
# DaTest = DaTest.apply(LablHtEncding.fit_transform)


name2re = open('kddcup_names.txt')

re2read = csv.reader(name2re)
read2list = list(re2read)
list2narray = numpy.array(read2list[0])

# print(DaSt.isnull().sum())

# clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

inp_tra, inp_test, oup_tra, oup_test = train_test_split(DaSt, labels, test_size=0.20)

# Scale Inputs for better final results
Sclr = StandardScaler()
Sclr.fit(inp_tra)

inp_tra = Sclr.transform(inp_tra)
inp_test = Sclr.transform(inp_test)

startTimePr1 = datetime.now()
# MLP initialiation and fit dataset
print('MLP classifier Normal in progress....')
NN_normal = MLPClassifier(hidden_layer_sizes=(41, 180, 23), max_iter=500, solver='sgd')
NN_normal.fit(inp_tra, oup_tra)

print('MLP classifier /w Early stopping in progress....')
startTimePr2 = datetime.now()
NN_EarlyStp = MLPClassifier(hidden_layer_sizes=(41, 180, 23), max_iter=500, solver='sgd', early_stopping=True)
NN_EarlyStp.fit(inp_tra, oup_tra)

startTimePr3 = datetime.now()
# Random Forest
print('Random Forest Classifier in progress.....')
RanForNor = RandomForestClassifier()
RanForNor.fit(inp_tra, oup_tra)

# Predict (Norm) for confusion matrix
preds_NN = NN_normal.predict(inp_test)
preds_NN_EaSp = NN_EarlyStp.predict(inp_test)
preds_RanForClas = RanForNor.predict(inp_test)



# Probability prediction for plotting
#training
prob_NN_EaSp_Tra = NN_EarlyStp.predict_proba(inp_tra)
prob_NN_Nor_Tra = NN_normal.predict_proba(inp_tra)
prob_RanFor_Tra = RanForNor.predict_proba(inp_tra)

#testing
NN_normalTest = NN_normal
NN_normalTest.fit(inp_test, oup_test)

NN_EarlyStpTest = NN_EarlyStp
NN_EarlyStpTest.fit(inp_test, oup_test)

RanForNorTest = RanForNor
RanForNorTest.fit(inp_test, oup_test)

prob_NN_EaSp_Test = NN_EarlyStpTest.predict_proba(inp_test)
prob_NN_Nor_Test = NN_normalTest.predict_proba(inp_test)
prob_RanFor_Test = RanForNorTest.predict_proba(inp_test)




# Score for individual Classifiers
txtRec = open("output.txt", "a")
print('Pred Neural Network', file=txtRec)
print((classification_report(oup_test, preds_NN, zero_division=1)), file=txtRec)
print('Time taken MLP', datetime.now() - startTimePr1)
print('--' * 30, file=txtRec)
print('Pred Neural Network /w Early Stopping', file=txtRec)
print((classification_report(oup_test, preds_NN_EaSp, zero_division=1)), file=txtRec)
print('Time taken MLP', datetime.now() - startTimePr3)
print('--' * 30, file=txtRec)
print('Pred Rand Forest Classifier', file=txtRec)
print((classification_report(oup_test, preds_RanForClas, zero_division=1)), file=txtRec)
print('Time taken RandomFC', datetime.now() - startTimePr2)
print(('#' * 50), file=txtRec)
print()
# storeMLP
# store
# storeNN


MLPRes01 = precision_score(oup_test, preds_NN, average='macro', zero_division=1)
MLPRes02 = recall_score(oup_test, preds_NN, average='macro', zero_division=1)
MLPRes03 = f1_score(oup_test, preds_NN, average='macro', zero_division=1)
MLPRes04 = accuracy_score(oup_test, preds_NN)
MLPRes05 = balanced_accuracy_score(oup_test, preds_NN)

# MLPESRes01 = precision_recall_fscore_support(oup_test, preds_NN)
MLPESRes01 = precision_score(oup_test, preds_NN_EaSp, average='macro', zero_division=1)
MLPESRes02 = recall_score(oup_test, preds_NN_EaSp, average='macro', zero_division=1)
MLPESRes03 = f1_score(oup_test, preds_NN_EaSp, average='macro', zero_division=1)
MLPESRes04 = accuracy_score(oup_test, preds_NN_EaSp)
MLPESRes05 = balanced_accuracy_score(oup_test, preds_NN_EaSp)

# RaFoClRes01 = precision_recall_fscore_support(oup_test, preds_NN)
RaFoClRes01 = precision_score(oup_test, preds_RanForClas, average='macro', zero_division=1)
RaFoClRes02 = recall_score(oup_test, preds_RanForClas, average='macro', zero_division=1)
RaFoClRes03 = f1_score(oup_test, preds_RanForClas, average='macro', zero_division=1)
RaFoClRes04 = accuracy_score(oup_test, preds_RanForClas)
RaFoClRes05 = balanced_accuracy_score(oup_test, preds_RanForClas)

# Tim, res = test.start(DaSt, list2narray, labels)
# res = round(res, 2)

MLPtime = datetime.now() - startTimePr1
MLPestime = datetime.now() - startTimePr2
RaFoClatime = datetime.now() - startTimePr3

print(' ' + '#' + '-------' * 22 + '#')
print(' ' + '|', '-' * 26, '| ', 'Precision Score', '  |  ', 'Recall Score', '  |  ''F1-Score', '  |  ',
      'Accuracy Score', '  |  ', 'Balanced Accuracy Score', '  |     ', 'Duration', ' ' * 3 + ' |')
print(' ' + '|', 'Multi-Layer Perceptron  ', ' ', '| ', ' ' * 5, round(MLPRes01 * 100), '%', ' ' * 15,
      round(MLPRes02 * 100), '%', ' ' * 10, round(MLPRes03 * 100), '%', ' ' * 11, round(MLPRes04 * 100), '%', ' ' * 19,
      round(MLPRes05 * 100), '%', ' ' * 13, MLPtime, '  |')
print(' ' + '|', 'MLP w/early Stopping    ', ' ', '| ', ' ' * 5, round(MLPESRes01 * 100), '%', ' ' * 15,
      round(MLPESRes02 * 100), '%', ' ' * 10, round(MLPESRes03 * 100), '%', ' ' * 11, round(MLPESRes04 * 100), '%',
      ' ' * 19, round(MLPESRes05 * 100), '%', ' ' * 13, MLPestime, '  |')
print(' ' + '|', 'Random Forest Classifier  ', '| ', ' ' * 5, round(RaFoClRes01 * 100), '%', ' ' * 15,
      round(RaFoClRes02 * 100), '%', ' ' * 10, round(RaFoClRes03 * 100), '%', ' ' * 11, round(RaFoClRes04 * 100), '%',
      ' ' * 19, round(RaFoClRes05 * 100), '%', ' ' * 13, RaFoClatime, '  |')
# print('|', 'Neural-Network ', '| ', ' ' * 5, 55.55, ' ' * 13, 55.55, ' ' * 9,55.55, ' ' * 10,55.55, ' ' * 18,res, ' ' * 16, Tim, ' ' * 2, '|')
print(' ' + '#' + '-------' * 22 + '#')
print()
print('Writting to "Output.txt"')
print()
print(' ' + '#' + '-------' * 22 + '#', file=txtRec)
print(' ' + '|', '-' * 26, '| ', 'Precision Score', '  |  ', 'Recall Score', '  |  ''F1-Score', '  |  ',
      'Accuracy Score', '  |  ', 'Balanced Accuracy Score', '  |     ', 'Duration', ' ' * 3 + ' |', file=txtRec)
print(' ' + '|', 'Multi-Layer Perceptron  ', ' ', '| ', ' ' * 5, round(MLPRes01 * 100), '%', ' ' * 15,
      round(MLPRes02 * 100), '%', ' ' * 10, round(MLPRes03 * 100), '%', ' ' * 11, round(MLPRes04 * 100), '%', ' ' * 19,
      round(MLPRes05 * 100), '%', ' ' * 13, MLPtime, '  |', file=txtRec)
print(' ' + '|', 'MLP w/early Stopping    ', ' ', '| ', ' ' * 5, round(MLPESRes01 * 100), '%', ' ' * 15,
      round(MLPESRes02 * 100), '%', ' ' * 10, round(MLPESRes03 * 100), '%', ' ' * 11, round(MLPESRes04 * 100), '%',
      ' ' * 19, round(MLPESRes05 * 100), '%', ' ' * 13, MLPestime, '  |', file=txtRec)
print(' ' + '|', 'Random Forest Classifier  ', '| ', ' ' * 5, round(RaFoClRes01 * 100), '%', ' ' * 15,
      round(RaFoClRes02 * 100), '%', ' ' * 10, round(RaFoClRes03 * 100), '%', ' ' * 11, round(RaFoClRes04 * 100), '%',
      ' ' * 19, round(RaFoClRes05 * 100), '%', ' ' * 13, RaFoClatime, '  |', file=txtRec)
# print('|', 'Neural-Network ', '| ', ' ' * 5, 55.55, ' ' * 13, 55.55, ' ' * 9,55.55, ' ' * 10,55.55, ' ' * 18,res, ' ' * 16, Tim, ' ' * 2, '|')
print(' ' + '#' + '-------' * 22 + '#', file=txtRec)

print('#' * 50)
print()
print('Total Time taken for Classifiers', datetime.now() - startTime)
print('Displaying figures...')
print('...for training..')
#training
skP.metrics.plot_roc(oup_tra, prob_NN_Nor_Tra, title='ROC(training): MLP')
skP.metrics.plot_roc(oup_tra, prob_NN_EaSp_Tra, title='ROC(training): MLP /w Early Stopping')
skP.metrics.plot_roc(oup_tra, prob_RanFor_Tra, title='ROC(training): Random Forest')

skP.metrics.plot_precision_recall(oup_tra, prob_NN_Nor_Tra, title='Precision-Recall Curve(training): MLP')
skP.metrics.plot_precision_recall(oup_tra, prob_NN_EaSp_Tra, title='Precision-Recall Curve(training): MLP /w Early Stopping')
skP.metrics.plot_precision_recall(oup_tra, prob_RanFor_Tra, title='Precision-Recall Curve:(training) Random Forest')
print('...for testing..')

#testing
skP.metrics.plot_roc(oup_test, prob_NN_Nor_Test, title='ROC(testing): MLP')
skP.metrics.plot_roc(oup_test, prob_NN_EaSp_Test, title='ROC(testing): MLP /w Early Stopping')
skP.metrics.plot_roc(oup_test, prob_RanFor_Test, title='ROC(testing): Random Forest')


skP.metrics.plot_precision_recall(oup_test, prob_NN_Nor_Test, title='Precision-Recall Curve(testing): MLP')
skP.metrics.plot_precision_recall(oup_test, prob_NN_EaSp_Test, title='Precision-Recall Curve(testing): MLP /w Early Stopping')
skP.metrics.plot_precision_recall(oup_test, prob_RanFor_Test, title='Precision-Recall Curve:(testing) Random Forest')

#print('Displaying estimators...')
#skP.estimators.plot_learning_curve(NN_EarlyStp, inp_tra, oup_tra, title='Learning curve: MLP /w Early Stopping', scoring='f1', average='macro')

plt.show()



txtRec.close()
print('Total Time taken for Program with plots open', datetime.now() - startTime)
