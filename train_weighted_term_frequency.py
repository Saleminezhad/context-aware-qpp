import pickle
import numpy as np 

def filter_data2(test_data_path, test_reslt_path):

    with open(test_reslt_path, 'r') as file:
        test_result = file.read()

    with open(test_data_path, 'r') as file:
        test_data = file.read()

    test_result = test_result.split("\n")
    test_data = test_data.split("\n")

    if test_result[-1] == '':
        test_result = test_result[:-1]

    if test_data[-1] == '':
        test_data = test_data[:-1]
    
    print(len(test_result), len(test_data))

    filtered_data = {}
    for index in range(len(test_data)):

        qid,qtext = test_data[index].split("\t")
        filtered_data[qid]={}

        temp = []
        for  j,i in enumerate(test_result[index].split("\t")[:-1]):
            if "##" not in i:
                temp.append(i)


        try:
            temp2 = []
            temp3 = test_data[index].split("\t")[1].split(" ")
            for  j,i in enumerate(temp3):
                if temp[j].split(" ")[0] in i:
                    x = temp3[j] + " "+ temp[j].split(" ")[1]
                    temp2.append(x)
        except:
            pass

        line_temp = ""
        for term in temp2:
            text = term.split(" ")[0]
            weight_value = int(float(term.split(" ")[1])*100)
            line_temp += text + " " +  str(weight_value) + " "

        filtered_data[qid]["qtext"] = qtext
        filtered_data[qid]["term_weight"] = line_temp

        # filtered_data.append(temp2)
    return filtered_data


pos_weights_path = "/DeepCT2/predictions/BertQPP_Query_pos_max9_e12/test_results.tsv"
neg_weights_path = "/DeepCT2/predictions/BertQPP_Query_neg_max9_e12/test_results.tsv"  
Bert_train_path = "/BERTQPP2/pklfiles/train_map.pkl"
deepct_test_path = "/DeepCT2/data/queries.train.small.tsv"
# saved file path for trainging bertqpp
Bert_train_path_deepct_one_weight_pos = "/BERTQPP/pklfiles/term_freq_main_100.pkl" 

# term_freq_main2 mean dataloader2
# v2 deepct query clean 
# v3 my query clean

filtered_data_pos = filter_data2(deepct_test_path, pos_weights_path)
filtered_data_neg = filter_data2(deepct_test_path, neg_weights_path)

with open(Bert_train_path, 'rb') as pickle_file:
    # Load the data from the pickle file
    Bert_train_data = pickle.load(pickle_file)
Bert_train_data

hist_tf = []
for id in Bert_train_data.keys():
    pos_terms = filtered_data_pos[id]["term_weight"].split(" ")
    neg_terms = filtered_data_neg[id]["term_weight"].split(" ")

    q_pos_new = ""
    q_neg_new = ""

    for i in range(int(len(pos_terms)/2)):

        w_pos = abs(int(pos_terms[2*i +1]))
        W_neg = abs(int(neg_terms[2*i +1]))

        tf_new = w_pos - W_neg
        if tf_new != 0:
            hist_tf.append(tf_new)
        if tf_new > 0:
            q_pos_new += (pos_terms[2*i] + " ")*tf_new + " "
            q_neg_new += neg_terms[2*i] + " "
        elif tf_new< 0:
            q_pos_new += pos_terms[2*i] + " "
            q_neg_new += (neg_terms[2*i] + " ")*abs(tf_new) + " "
        else:
            q_pos_new += pos_terms[2*i] + " "
            q_neg_new += neg_terms[2*i] + " "

    Bert_train_data[id]['qtext'] = q_pos_new
    Bert_train_data[id]['doc_text'] = q_pos_new

with open(Bert_train_path_deepct_one_weight_pos, 'wb') as pickle_file:
    pickle.dump(Bert_train_data, pickle_file)
