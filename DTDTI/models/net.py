import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
from models.transformer_net import Encoder
from models.protein_cnn import ProteinCNN


if torch.cuda.is_available():
    device = torch.device('cuda')


class DTDTInet(nn.Module):
    def __init__(self, max_length=1000, compound_feature=41, compound_graph_dim=64, compound_smiles_dim=4,
                 protein_dim=64, out_dim=2, dropout=0.2):
        super(DTDTInet, self).__init__()
        self.max_length = max_length
        self.compound_feature = compound_feature
        self.compound_graph_dim = compound_graph_dim
        self.compound_smiles_dim = compound_smiles_dim

        self.protein_dim = protein_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.compound_gcn1 = GCNConv(in_channels=compound_feature, out_channels=compound_feature)
        self.compound_gcn2 = GCNConv(in_channels=compound_feature, out_channels=compound_feature * 2)
        self.compound_gcn3 = GCNConv(in_channels=compound_feature * 2, out_channels=compound_feature * 4)

        self.compound_encoder = Encoder(n_layers=10, in_dim=1, embed_size=128, heads=4, forward_expansion=4,
                                        dropout=0.2)

        self.protein_embedding = nn.Linear(1, 64)
        self.protein_cnn = ProteinCNN(block_num=3, embedding_num=128)

        self.protein_encoder = Encoder(n_layers=10, in_dim=1, embed_size=128, heads=4, forward_expansion=4, dropout=0.2)

        self.fc_g1 = nn.Linear(compound_feature * 4, 1024)
        self.fc_g2 = nn.Linear(1024, compound_graph_dim)
        self.fc_g3 = nn.Linear(64 * 128, 128)

        self.fc_p1 = nn.Linear(64 * 128, 128)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(128 + 64 + 128 + 64, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        target = data.target

        compound_graph = self.compound_gcn1(x, edge_index)
        compound_graph = self.relu(compound_graph)
        compound_graph = self.compound_gcn2(compound_graph, edge_index)
        compound_graph = self.relu(compound_graph)
        compound_graph = self.compound_gcn3(compound_graph, edge_index)
        compound_graph = self.relu(compound_graph)
        compound_graph = global_max_pool(compound_graph, batch)

        compound_graph = self.relu(self.fc_g1(compound_graph))
        compound_graph = self.dropout(compound_graph)
        compound_graph = self.fc_g2(compound_graph)
        compound_graph = self.dropout(compound_graph)

        compound_transformer = torch.unsqueeze(compound_graph, -1)
        compound_transformer = self.compound_encoder(compound_transformer)
        compound_transformer = compound_transformer.view(-1, 64 * 128)
        compound_transformer = self.fc_g3(compound_transformer)
        compound = torch.cat((compound_graph, compound_transformer), dim=1)
        # compound = compound_graph

        protein_cnn = self.protein_cnn(target)

        protein_transformer = torch.unsqueeze(protein_cnn, -1)

        protein_transformer = self.protein_encoder(protein_transformer)
        protein_transformer = protein_transformer.view(-1, 64 * 128)
        protein_transformer = self.fc_p1(protein_transformer)

        protein = torch.cat((protein_transformer, protein_cnn), dim=1)
        # protein = protein_cnn

        x = torch.cat((compound, protein), dim=1)

        x = self.classifier(x)

        return x
