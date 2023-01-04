# Sustainable_docuemnts_scoring
This repository contains application for trainig a NLP model and use it for evaluating an document agians a predefined benchmarks. 
# Recomendation_system__training_v26.py
This module use an input directory of zip file as a source for documents (pdf files) and train a GloVe model to creat a Room for a specific industry or sector.
It uses a pretrained 300 dimensional pretraimed model as initiation(glove_6B_300d.txt). the initial pretrained mode collectedform: <br />
from https://nlp.stanford.edu/projects/glove <br />
the module also use a benchmark list for keeping benchmarks in the dictionary: <br />
**file_bench = 'rooted_bench.csv'** <br />
The module has several initial configuarion factors that all are sets in begining of the module. 
The trained model is saved periodically (after each 20 files processed) in:<br />
**trained_model='glove_dictionary_test.pkl'** <br />
Larger files are splitted for memory limtation. 
If for any reason the process is stoped, the module must be rerun and next run will autoamtically  process only input files that were not already processd during earlier trainning.
After completly read all input files a meeaage appears about completion of the process. the final traiend model will be :<br /> **glove_dictionary_test.pkl**<br />

# Recomendation_system_NLP_v26.py
This module used a trained model as room a benchmark list and a directory of  documents. the input document will evaluated agians the room and the benchmark list and creat a output 
file containg the level of the similarity of the input documents words to bencmarks. <br />
The benchmarks were catagorized weighted by help of SMEs. The catagorizes are used for grouping required similarity measures. the output will saved as CSV file:<br />
**sim_to_bench_file='sim_to_bench.csv'**<br />

The required functions are imported through:
**from Carlo_ngrams_tool.utilities_recommendation import** <br /> to import:
**utilityies_recomndation.py** <br /> It should be properly located to be accesible by main modules. there are a library of text processing functions that is imported by utility_recomendation.py as well. it is:<br />
**chunking_bforce_plus_space_add.py**<br />
It should be also accesible properly. the library contains all text processing functions. <br /> the directory structure is as folllow:

├── working_directory<br />
│   ├── Recomendation_system__training_v26.py<br />
│   ├── Recomendation_system_NLP_v26.py<br />
│   ├── glove_dictionary_test.pkl<br />
│   ├── rooted_benchmarks_11_07_2022.csv<br />
│   ├── bench-to-crawled-duck_unique.csv<br />
│   ├── Carlo_ngrams_tool<br />
│   │       ├── utilityies_recomndation.py<br />
│   │       ├── chunking_bforce_plus_space_add.py<br />

