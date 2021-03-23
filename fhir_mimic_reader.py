"""
Implementing Text GCN for FHIR NLP results


-  The implementation of Text GCN :
    Liang Yao, Chengsheng Mao, Yuan Luo. "Graph Convolutional Networks for Text Classification."
    In 33rd AAAI Conference on Artificial Intelligence (AAAI-19), 7370-7377

- For FHIR-EHR cases: https://www.sciencedirect.com/science/article/pii/S1532046419302291
    Hong et al., Developing a FHIR-based EHR phenotyping framework: A case study for identification of
        patients with obesity and multiple comorbidities from discharge summaries, JBI

"""


from pathlib import Path
import json
import fhirclient.models.bundle as bd
import networkx as nx
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import xmltodict
from collections import defaultdict

import pandas as pd

#ignore warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn


data_root = Path('fhir/obesity_datasets')



class FhirDoc:
    def __init__(self, entry_name, doc_src='i2b2'):
        self.entry_name = entry_name
        self.i2b2_txt_root = data_root / 'mimic/notes/total'
        self.i2b2_bundle_root = data_root / 'mimic/ObesityResourceBundle'

        with open(self.i2b2_bundle_root / entry_name) as h:
            self.doc_js = json.load(h)
            self.bundle = bd.Bundle(self.doc_js, strict=False)

        with open(self.i2b2_txt_root / entry_name[:-5]) as f_txt:
            self.text = f_txt.read()

    def get_all_references(self):
        reference_list = []
        for entry in self.doc_js['entry']:
            if entry['fullUrl'].startswith('Composition'):
                continue

            try:
                coding = entry['resource']['code']['coding'][0]
                reference_list.append("{}-{}".format(coding['system'], coding['code']))
            except KeyError:
                # TODO: include all cases. Currently excluding a lot
                pass

        return reference_list

    def get_text(self):
        return self.text

    def get_line_text(self):
        return self.text.replace('\n', ' ')


tags = ['train', 'test']

diseases = ['Asthma', 'CAD', 'CHF', 'Depression', 'Diabetes', 'GERD', 'Gallstones', 'Hypercholesterolemia',
            'Hypertension', 'Hypertriglyceridemia', 'OA', 'OSA', 'Obesity', 'PVD', 'Venous-Insufficiency'
            ]


def get_mimic_labels():
    df = pd.read_csv('data/obesity_datasets/mimic/GOLD.csv')
    df.rename(columns={'Unnamed: 0': 'doc_id'}, inplace=True)

    df_dict = df.to_dict('index')

    disease_doc_labels = {}  # key1: disease; key2: doc_id; value: GS label

    for disease in diseases:
        disease_doc_labels[disease] = {}

    train_test_split_dict = {}

    for idx in df_dict:
        doc_dict = df_dict[idx]
        doc_id = str(doc_dict['doc_id'])
        for disease in diseases:
            disease_doc_labels[disease][doc_id] = doc_dict[disease]

            if doc_dict['train'] == 1:
                train_test_split_dict[doc_id] = 'train'
            else:
                train_test_split_dict[doc_id] = 'test'

    return train_test_split_dict, disease_doc_labels


def prepare_mimic():
    i = 0
    dataset_name = 'mimic'
    train_test_dict, disease_doc_labels = get_mimic_labels()

    sentences = {}
    references = {}

    for disease in diseases:
        sentences[disease] = defaultdict(list)       # key: 'train'/'test', value: list of sentences
        references[disease] = defaultdict(list)       # key: 'train'/'test', value: list of sentences

    # read fhir resources
    for bundle_path in data_root.joinpath('mimic/ObesityResourceBundle').glob('*.json'):
        doc_name = bundle_path.name                  # 'REPORT1179.txt.json'
        doc = FhirDoc(doc_name)
        doc_id = doc_name.split('.')[0]       # 'REPORT1179.txt.json' -> '1179'
        tag = train_test_dict.get(doc_id)     # 'train' or 'test'

        if tag is None:
            print('Warning: No label found in  train_test_dict: {} -> {}'.format(tag, doc_id))
            continue

        for disease in diseases:
            if disease_doc_labels[disease].get(doc_id) is None:
                print('Warning: No label found in disease_doc_labels: {} -> {}'.format(disease, doc_id))
                continue

            sentences[disease][tag].append((doc_id, disease_doc_labels[disease][doc_id], doc.get_line_text()))
            references[disease][tag].append((doc_id, disease_doc_labels[disease][doc_id], " ".join(doc.get_all_references())))

        i += 1

    # print all diseases
    for disease in diseases:
        # write to text_gcn directory
        fo_data = open('data/{}_dt_{}.txt'.format(dataset_name, disease), 'w')
        fo_corpus = open('data/corpus/{}_dt_{}.txt'.format(dataset_name, disease), 'w')

        fo_r_data = open('data/{}_rt_{}.txt'.format(dataset_name, disease), 'w')
        fo_ref = open('data/corpus/{}_rt_{}.txt'.format(dataset_name, disease), 'w')

        for tag in tags:
            for doc_id, label, sent in sentences[disease][tag]:
                fo_data.write("{}\t{}\t{}\n".format(doc_id, tag, label))
                # fo_corpus.write("{}\n".format(sent[:min(len(sent), 6000)]))
                fo_corpus.write("{}\n".format(sent))

            for doc_id, label, ref in references[disease][tag]:
                fo_ref.write("{}\n".format(ref))
                fo_r_data.write("{}\t{}\t{}\n".format(doc_id, tag, label))

        fo_data.close()
        fo_corpus.close()

        fo_ref.close()
        fo_r_data.close()


def get_graph_stats():
    # corpus level graph
    G = nx.Graph()

    i = 0
    for bundle_path in data_root.joinpath('mimic/ObesityResourceBundle').glob('*.json'):
        doc_name = bundle_path.name
        doc = FhirDoc(doc_name)
        # add doc node `O()`
        G.add_node(doc_name)

        re_list = doc.get_all_references()
        G.add_edges_from([(doc_name, fhir_id) for fhir_id in re_list])

        i += 1

    print(nx.number_of_nodes(G), nx.number_of_edges(G))
    print(nx.density(G))


if __name__ == '__main__':
    # prepare_mimic()
    get_graph_stats()