# nlp2fhir-deep-learning
Integration of NLP2FHIR Representation with Deep Learning Models for EHR Phenotyping

# Prerequsites

- Processed texts as a FHIR Resource Bundle from [NLP2FHIR](https://github.com/BD2KOnFHIR/NLP2FHIR)
- Python 3.5 and above
- Tensorflow >= 1.4.0
- fhirclient
- pandas
- networkx
- nltk

# Dataset preparation
## Input files
For both the i2b2 and mimic datasets, the input file structure under `data/obesity_datasets/{$DATASET}/` (replace `{$DATASET}` to `i2b2` or `mimic`) should be as follows:
- Notes: 
  - i2b2: `obesity notes/REPORT{$NOTE_ID}.txt`
  - mimic: `notes/{$NOTE_ID}.txt` 
- Resource Bundle (the output directory of NLP2FHIR): `` 
  - i2b2: `ObesityResourceResourceBundle/REPORT{$NOTE_ID}.txt.json`
  - mimic: `ObesityResourceResourceBundle/{$NOTE_ID}.txt.json`
- Gold standard labels 
  - i2b2: `train_groundtruth.xml` and `test_groundtruth.xml` 
  - mimic: a csv file with columns of ${NOTE_ID} and all comobidities as 0 and 1, and the last 2 columns as the indicator of training and testing, also in 0 and 1.

# Run 

Run `python fhir_{$DATASET}_reader.py`

Run `python remove_words.py {$DATASET}`

Run `python build_graph.py {$DATASET}`

Run `python train_on_keras_all_com.py`



# References
The implementation of Text GCN :

- Liang Yao, Chengsheng Mao, Yuan Luo. "Graph Convolutional Networks for Text Classification."
    In 33rd AAAI Conference on Artificial Intelligence (AAAI-19), 7370-7377
     [[Paper](https://arxiv.org/abs/1809.05679)] [[Code](https://github.com/yao8839836/text_gcn)]
    - Require: Python 2.7 or 3.6, Tensorflow >= 1.4.0


For FHIR-EHR cases

-  Hong et al., Developing a FHIR-based EHR phenotyping framework: A case study for identification of
        patients with obesity and multiple comorbidities from discharge summaries, JBI  [[Paper](https://www.sciencedirect.com/science/article/pii/S1532046419302291)]


