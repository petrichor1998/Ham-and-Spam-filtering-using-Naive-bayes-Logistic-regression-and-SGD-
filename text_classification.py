import io
import os
import time
import pandas as pd
import re
import numpy as np
import argparse
# from sklearn import linear_model
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import ParameterGrid
start_time = time.time()

def get_files(path):
    doc_list = []
    file_list = os.listdir(path)
    for fname in file_list:
        f = io.open(path + "/" + fname, 'rb')
        f_text = f.read()
        f_text = str(f_text)
        f_text_list = re.findall(r'\w+', f_text)
        f_text_list = filter(keep_words, f_text_list)
        doc_list.append(f_text_list)
    return doc_list

def keep_words(el):
    if el.isdigit():
        return False
    else:
        return True
def find_unique(h_list, s_list):
    unique_words = set()
    for ls in h_list:
        #temp_l = set(ls)
        temp_l = set(filter(keep_words, ls))
        unique_words = unique_words.union(temp_l)
    for ls in s_list:
        #temp_l = set(ls)
        temp_l = set(filter(keep_words, ls))
        unique_words = unique_words.union(temp_l)
    return list(unique_words)

def BOW(h_list, s_list, unique_words):
    hdocs_freq_list = []
    sdocs_freq_list = []
    for doc in h_list:
        temp = dict(zip(list(unique_words), [0]*len(unique_words)))
        for word in doc:
            if word in temp.keys():
                temp[word] = temp[word] + 1
        hdocs_freq_list.append(temp)
    for doc in s_list:
        temp = dict(zip(list(unique_words), [0]*len(unique_words)))
        for word in doc:
            if word in temp.keys():
                temp[word] = temp[word] + 1
        sdocs_freq_list.append(temp)

    ham_df = pd.DataFrame(hdocs_freq_list)
    spam_df = pd.DataFrame(sdocs_freq_list)
    ham_df["Y"] = [1 for i in range(len(ham_df.index))]
    spam_df["Y"] = [0 for i in range(len(spam_df.index))]
    hs_df = pd.concat([ham_df, spam_df], ignore_index=True)
    return hs_df
def Bernoulli(h_list, s_list, unique_words):
    hdocs_hash_list = []
    sdocs_hash_list = []
    for doc in h_list:
        temp = dict(zip(list(unique_words), [0] * len(unique_words)))
        for word in doc:
            if word in temp.keys() and temp[word] != 1:
                temp[word] = temp[word] + 1
        hdocs_hash_list.append(temp)
    for doc in s_list:
        temp = dict(zip(list(unique_words), [0] * len(unique_words)))
        for word in doc:
            if word in temp.keys() and temp[word] != 1:
                temp[word] = temp[word] + 1
        sdocs_hash_list.append(temp)
    ham_df = pd.DataFrame(hdocs_hash_list)
    spam_df = pd.DataFrame(sdocs_hash_list)
    ham_df["Y"] = [1 for i in range(len(ham_df.index))]
    spam_df["Y"] = [0 for i in range(len(spam_df.index))]
    hs_df = pd.concat([ham_df, spam_df], ignore_index=True)
    return hs_df

def multiNB_trainer(bow_df):
    #prior_L[0] is Y = 1
    prior_l = [0,0]
    prior_l[0] = np.log2(np.true_divide(bow_df["Y"].sum(), len(bow_df.index)))
    prior_l[1] = np.log2(1 - np.true_divide(bow_df["Y"].sum(), len(bow_df.index)))
    prob_dict_h = dict(zip(list(bow_df.columns[:-1]), [0]*len(bow_df.columns[:-1])))
    prob_dict_s = dict(zip(list(bow_df.columns[:-1]), [0]*len(bow_df.columns[:-1])))
    prob_h = np.zeros(len(bow_df.columns[:-1]))
    prob_s = np.zeros(len(bow_df.columns[:-1]))
    #calculating conditional probabilities of the words
    dsum_h = 0
    dsum_s = 0
    for i, col in enumerate(bow_df.columns[:-1]):
        prob_h[i] = bow_df[col][bow_df["Y"] == 1].sum() + 1
        prob_s[i] = bow_df[col][bow_df["Y"] == 0].sum() + 1
        dsum_h = dsum_h + (bow_df[col][bow_df["Y"] == 1].sum() + 1)
        dsum_s = dsum_s + (bow_df[col][bow_df["Y"] == 0].sum() + 1)
    dsum_h = np.log2(dsum_h)
    dsum_s = np.log2(dsum_s)
    prob_h = np.log2(prob_h)
    prob_s = np.log2(prob_s)
    prob_h = prob_h - dsum_h
    prob_s = prob_s - dsum_s
    for i, word in enumerate(list(bow_df.columns[:-1])):
        prob_dict_h[word] = prob_h[i]
        prob_dict_s[word] = prob_s[i]
    return prob_dict_h, prob_dict_s, prior_l

def multiNB_label_maker(hw_probs, sw_probs, p_list, hl_test, sl_test, bow_df):
    #check conditional probability of the examples given Y  = 1
    Y = [1 for i in range(len(hl_test))]
    Y = Y + [0 for i in range(len(sl_test))]
    pred_Y = []
    for doc in hl_test + sl_test:
        hscore = p_list[0]
        sscore = p_list[1]
        for word in doc:
            if word in hw_probs:
                hscore = hscore + hw_probs[word]
            if word in sw_probs:
                sscore = sscore + sw_probs[word]
        if sscore > hscore:
            pred_Y.append(0)
        else:
            pred_Y.append(1)
    boolY = np.array(Y) == np.array(pred_Y)
    df = pd.DataFrame(zip(np.array(Y), np.array(pred_Y)), columns = ["Y","pred_Y"])
    TP = df["pred_Y"][df["Y"] == 1]
    TP = TP.values.sum()
    TN = df["pred_Y"][df["Y"] == 0]
    TN = len(TN.index) - TN.values.sum()
    FP = df["pred_Y"][df["Y"] == 0]
    FP = FP.values.sum()
    FN = df["pred_Y"][df["Y"] == 1]
    FN =len(FN.index) - FN.values.sum()
    accu = np.true_divide(TP+TN, TP+FP+FN+TN)
    prec = np.true_divide(TP, TP+FP)
    recall = np.true_divide(TP, TP+FN)
    F1 = 2*(recall * prec) / (recall + prec)
    return [accu, prec, recall, F1]

def biNB_trainer(ber_df):
    prior_l = [0, 0]
    prior_l[0] = np.log2(np.true_divide(ber_df["Y"].sum(), len(ber_df.index)))
    prior_l[1] = np.log2(1 - np.true_divide(ber_df["Y"].sum(), len(ber_df.index)))
    prob_dict_h = dict(zip(list(ber_df.columns[:-1]), [0] * len(ber_df.columns[:-1])))
    prob_dict_s = dict(zip(list(ber_df.columns[:-1]), [0] * len(ber_df.columns[:-1])))
    prob_h = np.zeros(len(ber_df.columns[:-1]))
    prob_s = np.zeros(len(ber_df.columns[:-1]))

    for i, col in enumerate(ber_df.columns[:-1]):
        prob_h[i] = ber_df[col][ber_df["Y"] == 1].sum() + 1
        prob_s[i] = ber_df[col][ber_df["Y"] == 0].sum() + 1
        # dsum_h = dsum_h + (ber_df[col][ber_df["Y"] == 1].sum() + 1)
        # dsum_s = dsum_s + (ber_df[col][ber_df["Y"] == 0].sum() + 1)
    dsum_h = np.log2(len(ber_df[ber_df["Y"] == 1].index) + 2)
    dsum_s = np.log2(len(ber_df[ber_df["Y"] == 0].index) + 2)
    prob_h = np.log2(prob_h)
    prob_s = np.log2(prob_s)
    prob_h = prob_h - dsum_h
    prob_s = prob_s - dsum_s
    for i, word in enumerate(list(ber_df.columns[:-1])):
        prob_dict_h[word] = prob_h[i]
        prob_dict_s[word] = prob_s[i]
    return prob_dict_h, prob_dict_s, prior_l

def biNB_label_maker(hw_probs, sw_probs, p_list, hl_test, sl_test, ber_df):
    #check conditional probability of the examples given Y  = 1
    Y = [1 for i in range(len(hl_test))]
    Y = Y + [0 for i in range(len(sl_test))]
    pred_Y = []
    for doc in hl_test + sl_test:
        hscore = p_list[0]
        sscore = p_list[1]
        for word in ber_df.columns[:-1]:
            if word in doc:
                hscore = hscore + hw_probs[word]
                sscore = sscore + sw_probs[word]
            else:
                hscore = hscore + (1 - hw_probs[word])
                sscore = sscore + (1 - sw_probs[word])
        if sscore > hscore:
            pred_Y.append(0)
        else:
            pred_Y.append(1)
    df = pd.DataFrame(zip(np.array(Y), np.array(pred_Y)), columns=["Y", "pred_Y"])
    TP = df["pred_Y"][df["Y"] == 1]
    TP = TP.values.sum()
    TN = df["pred_Y"][df["Y"] == 0]
    TN = len(TN.index) - TN.values.sum()
    FP = df["pred_Y"][df["Y"] == 0]
    FP = FP.values.sum()
    FN = df["pred_Y"][df["Y"] == 1]
    FN = len(FN.index) - FN.values.sum()
    #print(TP, TN, FP, FN)
    accu = np.true_divide(TP + TN, TP + FP + FN + TN)
    if TP + FP == 0:
        prec = 0
    else:
        prec = np.true_divide(TP, TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = np.true_divide(TP, TP + FN)
    if recall + prec == 0:
        F1 = 0
    else:
        F1 = np.true_divide(2 * (recall * prec), (recall + prec))
    return [accu, prec, recall, F1]

def log_reg_train(df, L, N):
    W0 = 0
    W = np.zeros(len(df.columns[:-1]))
    for i in range(500):
        if i % 100 == 0:
            print("{}th epoch".format(i))
        gradw = np.zeros(len(df.columns[:-1]))
        gradW0 = 0
        for i, row in df.iloc[:, :-1].iterrows():
            gradw = gradw + np.array(row) * (df["Y"][i] - np.true_divide(1, 1 + np.exp(W0 + (W * np.array(row)).sum()))) - L * W
            gradW0 = gradW0 + (df["Y"][i] - np.true_divide(1, 1 + np.exp(W0 + (W * np.array(row)).sum())))
        W = W + N * gradw
        W0 = W0 + N * gradW0
    return W, W0

def logreg_label_maker(W, W0, test_df, train_df):
    tecols = list(test_df.columns)
    #test_df[test_df.columns[test_df.columns.isin(cols)]]
    for col in tecols:
        if col not in train_df.columns:
            test_df = test_df.drop([col], axis = 1)
    pred_Y = np.zeros(len(test_df.iloc[:,:-1].index))
    c = []
    indexes = []
    for i, col in enumerate(train_df.columns):
        if col not in test_df.columns:
            indexes.append(i)
            c.append(col)
    train_df = train_df.drop(c, axis=1)
    W = np.delete(W, indexes)
    test_df[train_df.columns]
    for i, row in test_df.iloc[:, :-1].iterrows():
        pred_Y[i] = np.true_divide(1, 1 + np.exp(W0 + (W * np.array(row)).sum()))
    Y_df = pd.DataFrame(zip(np.array(test_df["Y"]), np.array(pred_Y)), columns=["Y", "pred_Y"])
    TP = Y_df["pred_Y"][Y_df["Y"] == 1]
    TP = TP.values.sum()
    TN = Y_df["pred_Y"][Y_df["Y"] == 0]
    TN = len(TN.index) - TN.values.sum()
    FP = Y_df["pred_Y"][Y_df["Y"] == 0]
    FP = FP.values.sum()
    FN = Y_df["pred_Y"][Y_df["Y"] == 1]
    FN = len(FN.index) - FN.values.sum()
    accu = np.true_divide(TP + TN, TP + FP + FN + TN)
    prec = np.true_divide(TP, TP + FP)
    recall = np.true_divide(TP, TP + FN)
    F1 = 2 * (recall * prec) / (recall + prec)
    return [accu, prec, recall, F1]

# def sgd_train_test(bow_df, bow_test_df):
#
#     X_train = np.array(bow_df.iloc[:, :-1])
#     Y_train = np.array(bow_df.iloc[:, -1])
#     X_test = np.array(bow_test_df.iloc[:, :-1])
#     Y_test = np.array(bow_test_df["Y"])
#
#     N = np.array([1e-2, 1e-1, 1e0, 1e1])
#     #N = np.array([1e-1])
#     njobs =np.array([-1])
#     epoch = np.array([200, 500])
#
#     tecols = list(bow_test_df.columns)
#     for col in tecols:
#         if col not in bow_df.columns:
#             bow_test_df = bow_test_df.drop([col], axis=1)
#
#     mod = linear_model.SGDClassifier()
#     grid = GridSearchCV(estimator=mod, param_grid=dict(alpha=N, n_jobs = njobs,max_iter = epoch))
#     grid.fit(X_train, Y_train)
#     print("Best score : ", grid.best_score_)
#     print("Best params :",  grid.best_params_)
#
#     pred_Y = grid.predict(bow_df[:,:-1].values)
#
#     Y_df = pd.DataFrame(zip(np.array(Y_test), np.array(pred_Y)), columns=["Y", "pred_Y"])
#     TP = Y_df["pred_Y"][Y_df["Y"] == 1]
#     TP = TP.values.sum()
#     TN = Y_df["pred_Y"][Y_df["Y"] == 0]
#     TN = len(TN.index) - TN.values.sum()
#     FP = Y_df["pred_Y"][Y_df["Y"] == 0]
#     FP = FP.values.sum()
#     FN = Y_df["pred_Y"][Y_df["Y"] == 1]
#     FN = len(FN.index) - FN.values.sum()
#     accu = np.true_divide(TP + TN, TP + FP + FN + TN)
#     prec = np.true_divide(TP, TP + FP)
#     recall = np.true_divide(TP, TP + FN)
#     F1 = 2 * (recall * prec) / (recall + prec)
#     return [accu, prec, recall, F1]

def main():
    parser = argparse.ArgumentParser(description="Runs the specified algorithm on the file path mentioned")
    parser.add_argument('-train_data', '--train_set', type=str)
    parser.add_argument('-test_data', '--test_set', type=str)

    arg = parser.parse_args()
    train_name = arg.train_set
    test_name = arg.test_set

# ham_train = "hw2_train/train/ham"
# spam_train = "hw2_train/train/spam"
# ham_test = "hw2_test/test/ham"
# spam_test = "hw2_test/test/spam"

    ham_train = train_name + "/ham"
    spam_train = train_name + "/spam"
    ham_test = test_name + "/ham"
    spam_test = test_name + "/spam"

    print("Getting a list of all words in the spam and ham documents....")
    #get a list of list of all words in each document in the ham and spam folder
    ham_list_train = get_files(ham_train)
    spam_list_train = get_files(spam_train)
    ham_list_test = get_files(ham_test)
    spam_list_test = get_files(spam_test)



    #find unique words in the ham and spam files
    unique_words_list = find_unique(ham_list_train, spam_list_train)
    #unique_words_list = find_unique(ham_list_test, spam_list_test)
    #get the ham and spam BOW model
    print("Creating BOW model...")
    hs_bow_df = BOW(ham_list_train, spam_list_train, unique_words_list)
    #get the ham and spam bernoulli model
    print("Creating bernoulli model...")
    hs_ber_df = Bernoulli(ham_list_train, spam_list_train, unique_words_list)


    print("Naive Bayes with Bag of Words Model...")

    ham_word_probs1, spam_word_probs1, prior_list1 = multiNB_trainer(hs_bow_df)

    pf_measures1 = multiNB_label_maker(ham_word_probs1, spam_word_probs1, prior_list1, ham_list_test, spam_list_test, hs_bow_df)
    print("Accuracy = {}, Precision = {}, Recall = {}, F1 score= {}".format(pf_measures1[0], pf_measures1[1], pf_measures1[2], pf_measures1[3]))
    print("Naive Bayes with Bernoulli Model...")

    ham_word_probs2, spam_word_probs2, prior_list2 = biNB_trainer(hs_ber_df)

    pf_measures2 = biNB_label_maker(ham_word_probs2, spam_word_probs2, prior_list2, ham_list_test, spam_list_test, hs_ber_df)
    print("Accuracy = {}, Precision = {}, Recall = {}, F1 score= {}".format(pf_measures2[0], pf_measures2[1], pf_measures2[2], pf_measures2[3]))

    print("Logistic Regression with Bow Model training...")
    Weights_bow, bias_bow = log_reg_train(hs_bow_df, 0.05, 0.1)
    print("Logistic Regression with Bernoulli Model training...")
    Weights_ber, bias_ber = log_reg_train(hs_ber_df, 0.05, 0.1)

    unique_words_list = find_unique(ham_list_test, spam_list_test)

    hs_bow_test_df = BOW(ham_list_test, spam_list_test, unique_words_list)

    hs_ber_test_df = Bernoulli(ham_list_test, spam_list_test, unique_words_list)
    print("Logistic Regression with Bow Model testing...")
    pf_measures3 = logreg_label_maker(Weights_bow, bias_bow, hs_bow_test_df, hs_bow_df)
    print("Accuracy = {}, Precision = {}, Recall = {}, F1 score= {}".format(pf_measures3[0], pf_measures3[1], pf_measures3[2], pf_measures3[3]))

    print("Logistic Regression with Bernoulli Model testing...")
    pf_measures4 = logreg_label_maker(Weights_ber, bias_ber, hs_ber_test_df, hs_ber_df)
    print("Accuracy = {}, Precision = {}, Recall = {}, F1 score= {}".format(pf_measures4[0], pf_measures4[1], pf_measures4[2], pf_measures4[3]))

    # print("SGD on bow...")
    # #pf_measures5 = sgd_train_test(hs_bow_df, hs_bow_test_df)
    # print("Accuracy = {}, Precision = {}, Recall = {}, F1 score= {}".format(pf_measures5[0], pf_measures5[1], pf_measures5[2], pf_measures5[3]) )
    #
    # print("SGD on bernoulli model...")
    # pf_measures6 = sgd_train_test(hs_ber_df, hs_ber_test_df)
    # print("Accuracy = {}, Precision = {}, Recall = {}, F1 score= {}".format(pf_measures6[0], pf_measures6[1], pf_measures6[2], pf_measures6[3]))

    print("time taken : ", time.time() - start_time)


main()
