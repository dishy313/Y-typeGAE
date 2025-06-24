import torch
import os, json
import re
import scipy.sparse as sp
import args
from preprocessing import *

def get_feature_embedding(net, nodeParm):
    net.eval()
    with torch.no_grad():
        feat_embedding = net.module.encoder(nodeParm)
    return feat_embedding

def make_graph_data(inputfile, model_CNNAE, device_info):
    features = []
    adjacent_arrays = []

    for itemJ in os.scandir(inputfile):
        if not itemJ.is_file():
            continue

        node_features = []
        adj_ori = torch.zeros(args.default_node_dim,args.default_node_dim)
        tmpids = []
        tmpChildren = {}

        with open(itemJ.path, "r", encoding='utf-8') as f:

            dict = json.loads(f.read())
            # read faces
            if 'faces' in dict.keys():
                for face in dict['faces']:
                    tmp = []
                    flag = True
                    if 'id' in face.keys():
                        if face['id'] not in tmpids:
                            tmpids.append(face['id'])
                        else:
                            flag = False
                    if flag:
                        if 'type' in face.keys():
                            pattern = re.compile(r'\d+')
                            nums = pattern.findall(face['type'])
                            typenum = int(nums[0])
                            if typenum == 10:
                                tmp.extend([-0.4])
                            elif typenum == 11:
                                tmp.extend([-0.6])
                            elif typenum == 12:
                                tmp.extend([-0.8])
                            elif typenum == 13:
                                tmp.extend([0.2])
                            elif typenum == 14:
                                tmp.extend([0.2])
                            elif typenum == 15:
                                tmp.extend([0.3])
                            elif typenum == 16:
                                tmp.extend([-1])
                            elif typenum == 17:
                                tmp.extend([0.4])
                            elif typenum == 18:
                                tmp.extend([0.4])
                            elif typenum == 19:
                                tmp.extend([0.4])
                        if 'locationRatio' in face.keys():
                            tmp.extend(face['locationRatio'])
                        if 'ratio' in face.keys():
                            tmp.extend(face['ratio'])
                        if 'locRatioAx' in face.keys():
                            tmp.extend(face['locRatioAx'])
                        if 'XDirection' in face.keys():
                            tmp.extend(face['XDirection'])
                        if 'YDirection' in face.keys():
                            tmp.extend(face['YDirection'])
                        if 'uvPoints' in face.keys() and 'uvNomals' in face.keys():
                            pt_feat_list = face['uvPoints']
                            norm_feat_list = face['uvNomals']

                            pt_feat = torch.tensor(pt_feat_list).to(device_info)
                            norm_feat = torch.tensor(norm_feat_list).to(device_info)

                            feat = torch.cat([pt_feat, norm_feat], 2).permute(2,0,1)
                            feat = feat.view(1, feat.shape[0], feat.shape[1], feat.shape[2])
                            feat_embedding = get_feature_embedding(model_CNNAE, feat)
                            feat_embedding = feat_embedding.view(feat_embedding.shape[-1])
                            feat_embedding = feat_embedding.tolist()
                            tmp.extend(feat_embedding)

                        if 'children' in face.keys():
                            tmpChildren[face['id']] = []
                            for child in face['children']:
                                if child not in tmpChildren[face['id']]:
                                    tmpChildren[face['id']].append(child)
                        node_features.append(tmp)

            # read edges
            if 'edges' in dict.keys():
                for edge in dict['edges']:
                    tmp = []
                    flag = True
                    if 'id' in edge.keys():
                        if edge['id'] not in tmpids:
                            tmpids.append(edge['id'])
                        else:
                            flag = False
                    if flag:
                        if 'type' in edge.keys():
                            pattern = re.compile(r'\d+')
                            nums = pattern.findall(edge['type'])
                            typenum = int(nums[0])
                            if typenum == 20:
                                tmp.extend([0.1])
                            elif typenum == 21:
                                tmp.extend([0.3])
                            elif typenum == 22:
                                tmp.extend([0.5])
                            elif typenum == 23:
                                tmp.extend([0.6])
                            elif typenum == 24:
                                tmp.extend([0.6])
                            elif typenum == 25:
                                tmp.extend([0.7])
                            elif typenum == 26:
                                tmp.extend([0.7])
                            elif typenum == 27:
                                tmp.extend([0.8])
                        if 'locationRatio' in edge.keys():
                            tmp.extend(edge['locationRatio'])
                        # if 'areaRatio' in edge.keys():
                        #     tmp.extend([edge['areaRatio']])
                        if 'ratio' in edge.keys():
                            tmp.extend(edge['ratio'])
                        if 'locRatioAx' in edge.keys():
                            if type(edge['locRatioAx']).__name__ == 'list':
                                tmp.extend(edge['locRatioAx'])
                            else:
                                if 'uvPoints' in edge.keys():
                                    pt_feat_list = edge['uvPoints']
                                    tmp.extend(pt_feat_list[0])
                        if 'XDirection' in edge.keys():
                            tmp.extend(edge['XDirection'])
                        if 'YDirection' in edge.keys():
                            tmp.extend(edge['YDirection'])
                        if 'uvPoints' in edge.keys() and 'uvNomals' in edge.keys():
                            pt_feat_list = edge['uvPoints']
                            norm_feat_list = edge['uvNomals']

                            pt_feat = torch.tensor(pt_feat_list).permute(1, 0).to(device_info)
                            norm_feat = torch.tensor(norm_feat_list).permute(1, 0).to(device_info)
                            pt_feat = pt_feat.view(3, args.sample_num, 1)
                            norm_feat = norm_feat.view(3, args.sample_num, 1)

                            feat = torch.cat([pt_feat, norm_feat], 0)
                            feat = feat.expand(6, args.sample_num, args.sample_num)

                            feat = feat.view(1, feat.shape[0], feat.shape[1], feat.shape[2])
                            feat_embedding = get_feature_embedding(model_CNNAE, feat)
                            feat_embedding = feat_embedding.view(feat_embedding.shape[-1])
                            feat_embedding = feat_embedding.tolist()
                            tmp.extend(feat_embedding)

                        if 'children' in edge.keys():
                            tmpChildren[edge['id']] = []
                            for child in edge['children']:
                                if child not in tmpChildren[edge['id']]:
                                    tmpChildren[edge['id']].append(child)
                        node_features.append(tmp)

            # read vertices
            if 'vertices' in dict.keys():
                for vertex in dict['vertices']:
                    tmp = []
                    flag = True
                    if 'id' in vertex.keys():
                        if vertex['id'] not in tmpids:
                            tmpids.append(vertex['id'])
                        else:
                            flag = False
                    if flag:
                        if 'type' in vertex.keys():
                            pattern = re.compile(r'\d+')
                            nums = pattern.findall(vertex['type'])
                            typenum = int(nums[0])
                            if typenum == 30:
                                tmp.extend([1])
                            elif typenum == 31:
                                tmp.extend([1])
                            elif typenum == 32:
                                tmp.extend([1])
                        if 'locationRatio' in vertex.keys():
                            tmp.extend(vertex['locationRatio'])
                        if 'ratio' in vertex.keys():
                            tmp.extend(vertex['ratio'])
                        if 'locRatioAx' in vertex.keys():
                            tmp.extend(vertex['locRatioAx'])
                        if 'XDirection' in vertex.keys():
                            tmp.extend(vertex['XDirection'])
                        if 'YDirection' in vertex.keys():
                            tmp.extend(vertex['YDirection'])
                        if 'uvPoints' in face.keys() and 'uvNomals' in face.keys():
                            pt_feat_list = vertex['uvPoints']
                            norm_feat_list = vertex['uvNomals']

                            pt_feat = torch.tensor(pt_feat_list).to(device_info)
                            norm_feat = torch.tensor(norm_feat_list).to(device_info)
                            pt_feat = pt_feat.view(3, 1, 1)
                            norm_feat = norm_feat.view(3, 1, 1)
                            feat = torch.cat([pt_feat, norm_feat], 0)
                            feat = feat.expand(6, args.sample_num, args.sample_num)

                            feat = feat.view(1, feat.shape[0], feat.shape[1], feat.shape[2])
                            feat_embedding = get_feature_embedding(model_CNNAE, feat)
                            feat_embedding = feat_embedding.view(feat_embedding.shape[-1])
                            feat_embedding = feat_embedding.tolist()
                            tmp.extend(feat_embedding)

                        if 'children' in vertex.keys():
                            tmpChildren[vertex['id']] = []
                            for child in vertex['children']:
                                if child not in tmpChildren[vertex['id']]:
                                    tmpChildren[vertex['id']].append(child)
                        node_features.append(tmp)

        for obj in tmpChildren.keys():
            firNum = tmpids.index(obj)
            for child in tmpChildren[obj]:
                if child in tmpids:
                    secNum = tmpids.index(child)
                    adj_ori[firNum][secNum] = 1
                    adj_ori[secNum][firNum] = 1
                else:
                    loopIndex = 0
                    if child.find('T') > 0:
                        loopIndex = 2
                    elif child.find('F') > 0:
                        loopIndex = 1
                    if loopIndex > 0:
                        tmpPos = child.find('-')
                        if tmpPos > 0:
                            loopId = int(child[tmpPos + 1:])
                            childName = int(child[:tmpPos - 1])
                            if childName in tmpids:
                                secNum = tmpids.index(childName)
                                adj_ori[firNum][secNum] = loopId + 1# + 0.5 * loopIndex
                                adj_ori[secNum][firNum] = loopId + 1# + 0.5 * loopIndex

        ori_len = len(node_features)
        for i in range(ori_len):
            adj_ori[i][i] = 1

        for i in range(ori_len, args.default_node_dim):
            node_features.append([0] * len(node_features[0]))
        feat_input = torch.tensor(node_features)
        feat_input = feat_input.to(device_info)
        features.append(feat_input)
        adjtmp = adj_ori.tolist()
        adj_input = adj_ori.to(device_info)

        adjacent_arrays.append(adj_input)

    return features, adjacent_arrays



class MyDataSet:
    def __init__(self, features, adjacent_arrays):
        self.features = features
        self.adjacent_arrays = adjacent_arrays

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.adjacent_arrays[idx]
