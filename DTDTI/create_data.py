import pandas as pd
import numpy as np
import os
import json, pickle

from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *


pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4


def atom_features(atom):
    """Generate atom features including atom symbol(17),degree(7),formal charge(1),
    radical electrons(1),hybridization(6),aromatic(1),hydrogen atoms attached(5),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']  # 17-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2,
                         'other']  # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
              one_of_k_encoding(atom.GetDegree(), degree) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [
                  atom.GetIsAromatic()]  # 17+7+2+6+1=33

    results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                              [0, 1, 2, 3, 4])  # 33+5=38
    try:
        results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    except:
        results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 38+3 =41
    return np.array(results)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


VOCAB_PROTEIN = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                 "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                 "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                 "U": 19, "T": 20, "W": 21,
                 "V": 22, "Y": 23, "X": 24,
                 "Z": 25}


def seqs2int(target):
    return [VOCAB_PROTEIN[s] for s in target]


def protein_to_vec(type):
    proteins = []
    max_length = 1000
    filename = 'dataset/human/processed/data_' + type + '.csv'
    N = len(open(filename).readlines())
    df = pd.read_csv(filename)
    for i, row in df.iterrows():
        print('/'.join(map(str, [i + 1, N - 1])))
        sequence = row['target_sequence']
        target = seqs2int(sequence)
        if len(target) < max_length:
            target = np.pad(target, (0, max_length - len(target)))
        else:
            target = target[:max_length]
        proteins.append(target)

    return proteins

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)

dataset = 'human'

compound_iso_smiles = []
for dt_name in ['human', 'celegans']:
    opts = ['train', 'val', 'test']
    for opt in opts:
        df = pd.read_csv('dataset/' + dataset + '/processed/data_' + opt + '.csv')
        compound_iso_smiles += list(df['compound_iso_smiles'])
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g

# convert to PyTorch data format
processed_data_file_train = 'dataset/' + dataset + '/processed/_train.pt'
processed_data_file_val = 'dataset/' + dataset + '/processed/_val.pt'
processed_data_file_test = 'dataset/' + dataset + '/processed/_test.pt'
if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
    df = pd.read_csv('dataset/' + dataset + '/processed/data_train.csv')
    train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
        df['affinity'])
    # XT = [seq_cat(t) for t in train_prots]
    XT = protein_to_vec('train')
    train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)

    df = pd.read_csv('dataset/' + dataset + '/processed/data_val.csv')
    val_drugs, val_prots, val_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])
    # XT = [seq_cat(t) for t in val_prots]
    XT = protein_to_vec('val')
    val_drugs, val_prots, val_Y = np.asarray(val_drugs), np.asarray(XT), np.asarray(val_Y)

    df = pd.read_csv('dataset/' + dataset + '/processed/data_test.csv')
    test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])
    # XT = [seq_cat(t) for t in test_prots]
    XT = protein_to_vec('test')
    test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)

    # make data PyTorch Geometric ready
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(root='dataset/' + dataset + '/processed/', dataset=dataset + '_train', xd=train_drugs,
                                xt=train_prots, y=train_Y, smile_graph=smile_graph)
    print('preparing ', dataset + '_train.pt in pytorch format!')
    val_data = TestbedDataset(root='dataset/' + dataset + '/processed/', dataset=dataset + '_val', xd=val_drugs,
                              xt=val_prots, y=val_Y, smile_graph=smile_graph)
    print('preparing ', dataset + '_test.pt in pytorch format!')
    test_data = TestbedDataset(root='dataset/' + dataset + '/processed/', dataset=dataset + '_test', xd=test_drugs,
                               xt=test_prots, y=test_Y, smile_graph=smile_graph)
    print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')
else:
    print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')
