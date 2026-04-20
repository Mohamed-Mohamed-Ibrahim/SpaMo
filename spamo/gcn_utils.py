import torch
import numpy as np
import torch.nn as nn
import pdb
import math
import copy


class Graph:
    """The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self, layout='custom', strategy='uniform', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        # 'body', 'left', 'right', 'mouth', 'face'
        # if layout == 'custom_hand21':
        if layout == 'left' or layout == 'right':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [0, 5],
                [5, 6],
                [6, 7],
                [7, 8],
                [0, 9],
                [9, 10],
                [10, 11],
                [11, 12],
                [0, 13],
                [13, 14],
                [14, 15],
                [15, 16],
                [0, 17],
                [17, 18],
                [18, 19],
                [19, 20],
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0
        
        elif layout == 'body':
            self.num_node = 9
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [3, 5],
                [5, 7],
                [4, 6],
                [6, 8],
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0
            
        elif layout == 'face_all':
            self.num_node = 9 + 8 + 1
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [[i, i + 1] for i in range(9 - 1)] + \
                             [[i, i + 1] for i in range(9, 9 + 8 - 1)] + \
                             [[9 + 8 - 1, 9]] + \
                             [[17, i] for i in range(17)]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = self.num_node - 1
            
        elif layout == 'mediapipe_69':
            self.num_node = 69
            self_link = [(i, i) for i in range(self.num_node)]

            # -------------------------------
            # BODY (0-10) based on MMPose slicing
            # 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar
            # 5:LSho, 6:RSho, 7:LElb, 8:RElb, 9:LWri, 10:RWri
            # -------------------------------
            body_edges = [
                (0, 1), (1, 3),          # Nose -> LEye -> LEar
                (0, 2), (2, 4),          # Nose -> REye -> REar
                (0, 5), (5, 7), (7, 9),  # Nose -> LShoulder -> LElbow -> LWrist
                (0, 6), (6, 8), (8, 10), # Nose -> RShoulder -> RElbow -> RWrist
                (5, 6)                   # LShoulder -> RShoulder
            ]

            # -------------------------------
            # FACE (11-26) 
            # -------------------------------
            face_offset = 11
            face_edges = [(i, i+1) for i in range(face_offset, face_offset+15)]
            face_edges.append((0, face_offset)) # Connect Nose to the Face chain

            # -------------------------------
            # LEFT HAND (27-47)
            # -------------------------------
            lh_offset = 27
            left_hand_edges = [
                (lh_offset+0, lh_offset+1),(lh_offset+1, lh_offset+2),(lh_offset+2, lh_offset+3),(lh_offset+3, lh_offset+4),
                (lh_offset+0, lh_offset+5),(lh_offset+5, lh_offset+6),(lh_offset+6, lh_offset+7),(lh_offset+7, lh_offset+8),
                (lh_offset+0, lh_offset+9),(lh_offset+9, lh_offset+10),(lh_offset+10, lh_offset+11),(lh_offset+11, lh_offset+12),
                (lh_offset+0, lh_offset+13),(lh_offset+13, lh_offset+14),(lh_offset+14, lh_offset+15),(lh_offset+15, lh_offset+16),
                (lh_offset+0, lh_offset+17),(lh_offset+17, lh_offset+18),(lh_offset+18, lh_offset+19),(lh_offset+19, lh_offset+20),
            ]

            # -------------------------------
            # RIGHT HAND (48-68)
            # -------------------------------
            rh_offset = 48
            right_hand_edges = [
                (rh_offset+0, rh_offset+1),(rh_offset+1, rh_offset+2),(rh_offset+2, rh_offset+3),(rh_offset+3, rh_offset+4),
                (rh_offset+0, rh_offset+5),(rh_offset+5, rh_offset+6),(rh_offset+6, rh_offset+7),(rh_offset+7, rh_offset+8),
                (rh_offset+0, rh_offset+9),(rh_offset+9, rh_offset+10),(rh_offset+10, rh_offset+11),(rh_offset+11, rh_offset+12),
                (rh_offset+0, rh_offset+13),(rh_offset+13, rh_offset+14),(rh_offset+14, rh_offset+15),(rh_offset+15, rh_offset+16),
                (rh_offset+0, rh_offset+17),(rh_offset+17, rh_offset+18),(rh_offset+18, rh_offset+19),(rh_offset+19, rh_offset+20),
            ]

            # -------------------------------
            # CONNECT HANDS TO BODY
            # -------------------------------
            connect_edges = [
                (9, lh_offset+0),   # Left Wrist (9) -> Left Hand Root (27)
                (10, rh_offset+0)   # Right Wrist (10) -> Right Hand Root (48)
            ]

            neighbor_link = (
                body_edges +
                face_edges +
                left_hand_edges +
                right_hand_edges +
                connect_edges
            )

            self.edge = self_link + neighbor_link
            self.center = 0
        else:
            raise ValueError(f"Unknown layout: '{layout}'. "f"Valid: left, right, body, face_all, mediapipe_69")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if (
                                self.hop_dis[j, self.center]
                                == self.hop_dis[i, self.center]
                            ):
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif (
                                self.hop_dis[j, self.center]
                                > self.hop_dis[i, self.center]
                            ):
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD
