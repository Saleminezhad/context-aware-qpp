import pandas as pd
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
############################################################

all_query_path = "DeepCT2/data/diamond.tsv"


#############################################################


def labeling(sentence1, sentence2):

    # Example sentences
    # easy query
    # sentence1 = "what kind of animals are in grasslands?" 
    # hard query
    # sentence2 = "Tropical grassland animals (which do not all occur in the same area) include moles, hyenas, and phants."

    # Remove punctuation from both sentences
    translator = str.maketrans('', '', string.punctuation)
    sentence1_clean = sentence1.translate(translator)
    sentence2_clean = sentence2.translate(translator)

    # Tokenize cleaned sentences
    words1 = word_tokenize(sentence1_clean)
    words2 = word_tokenize(sentence2_clean)

    stop_words = set(stopwords.words('english'))

    words1 = [word for word in words1 if word not in stop_words]
    words2 = [word for word in words2 if word not in stop_words]

    # Initialize a stemmer
    stemmer = PorterStemmer()

    # Create a mapping of stemmed words to their original forms


    stemmed_to_original1 = {}
    final_words1 = []
    # Stem words and create the mapping
    for word in words1:
        stemmed_word = stemmer.stem(word.lower())
        final_words1.append(stemmed_word)
        if stemmed_word not in stemmed_to_original1:
            stemmed_to_original1[stemmed_word] = [word]
        else:
            stemmed_to_original1[stemmed_word].append(word)

    final_words2 = []
    stemmed_to_original2 = {}
    # Stem words and create the mapping
    for word in words2:
        stemmed_word = stemmer.stem(word.lower())
        final_words2.append(stemmed_word)
        if stemmed_word not in stemmed_to_original2:
            stemmed_to_original2[stemmed_word] = [word]
        else:
            stemmed_to_original2[stemmed_word].append(word)


    # Now you can compare words and refer back to the original forms if needed

    ############

    labels1 = {}

    for word in final_words1:
        if word not in final_words2:
            labels1[stemmed_to_original1[word][0]] = 1
        else:
            pass

    ############
    labels2 = {}

    for word in final_words2:
        if word not in final_words1:
            labels2[stemmed_to_original2[word][0]] = 1
                
        else:
            pass

    return labels1,labels2

col_names = ['qid', 'q_orig','MRR10_orig', 'q_var','MRR10_var']
all_query = pd.read_csv(all_query_path, sep='\t', names=col_names)

data_1_0_msmarco = all_query.loc[(all_query["MRR10_orig"] <= 0.15)]
#print("length of this dataset is: ", len(data_1_0_msmarco))

data_lines_easy = []
data_lines_hard = []

for index in range(len(data_1_0_msmarco)):
    
    qid = data_1_0_msmarco.iloc[index]["qid"]
    q_easy = data_1_0_msmarco.iloc[index]["q_var"]
    q_hard = data_1_0_msmarco.iloc[index]["q_orig"]

    labels1,labels2 = labeling(q_easy, q_hard)

    json_data_line_hard = {"query": data_1_0_msmarco.iloc[index]["q_orig"], "term_recall": labels2, "doc": {"position": "1", "id": str(data_1_0_msmarco.iloc[index]["qid"]), "title": data_1_0_msmarco.iloc[index]["q_orig"]}}
    json_data_line_easy = {"query": data_1_0_msmarco.iloc[index]["q_var"], "term_recall": labels1, "doc": {"position": "1", "id": str(data_1_0_msmarco.iloc[index]["qid"]), "title": data_1_0_msmarco.iloc[index]["q_var"]}}
    
    data_lines_easy.append(json_data_line_easy)
    data_lines_hard.append(json_data_line_hard)


# Define the output JSON file path
output_file_path_easy = "DeepCT2/data/" + 'data_label_neg_015.json'
output_file_path_hard = "DeepCT2/data/" + 'data_label_pos_015.json'

# Write each data line as a separate JSON object on a single line
with open(output_file_path_easy, 'w') as json_file:
    for line in data_lines_easy:
        json.dump(line, json_file)
        json_file.write('\n')

with open(output_file_path_hard, 'w') as json_file:
    for line in data_lines_hard:
        json.dump(line, json_file)
        json_file.write('\n')