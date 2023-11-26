from Reuters import Reuters
from MODELS import MODELS
from Genetic import Genetic

# User Input
print('Insert the path in which the folder Data of the project is located.\nExample: /Users/joaovalerio/Documents/MAI UPC/2 Semester/URL/W1')
file_path = input('-> ')

# Open file
f = open(file_path + '/Data/Results.txt', "w")
f.write("Unsupervised and Reinforcement Learning\nPractical Work 1: An extended version of the k-means method for overlapping clustering\nJoão Valério\njoao.agostinho@estudiantat.upc.edu\n09/05/2023\n\n")

# Get Reuters preprocessed data
reuters_1_lsa, reuters_2_lsa, reuters_3_lsa, reuters_1_tags, reuters_2_tags, reuters_3_tags = Reuters(file_path + '/Data/reuters21578', f).preprocess_reuters()

# Run all models on Reuters
MODELS(f).run_all_models_on_Reuters(reuters_1_lsa, reuters_2_lsa, reuters_3_lsa, reuters_1_tags, reuters_2_tags, reuters_3_tags, k=10, number_of_runs=1)

# Get the Genetics preprocessed data
X, labels = Genetic(f).preprocessDataset(file_path + '/Data/Yeast/yeast.arff')

# Run all models on Genetics
MODELS(f).run_all_models_on_Genetic(X, labels, k=14, number_of_runs=1)

# Close file
f.close()