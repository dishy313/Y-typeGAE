import torch
import os, json
import args

def make_graph_data(inputfile, device_info, batch_size = 1):
    features = []

    node_features = []
    tmpids = []

    for itemJ in os.scandir(inputfile):
        if not itemJ.is_file():
            continue

        with open(itemJ.path, "r", encoding='utf-8') as f:
            dict = json.loads(f.read())
            # read faces
            if 'faces' in dict.keys():
                for face in dict['faces']:
                    flag = True
                    if 'id' in face.keys():
                        if face['id'] not in tmpids:
                            tmpids.append(face['id'])
                        else:
                            flag = False
                    if flag:
                        if 'uvPoints' in face.keys() and 'uvNomals' in face.keys():
                            pt_feat_list = face['uvPoints']
                            norm_feat_list = face['uvNomals']
                            pt_feat = torch.tensor(pt_feat_list).to(device_info)
                            norm_feat = torch.tensor(norm_feat_list).to(device_info)
                            feat = torch.cat([pt_feat, norm_feat], 2).permute(2,0,1)
                        node_features.append(feat)

            # read edges
            if 'edges' in dict.keys():
                for edge in dict['edges']:
                    flag = True
                    if 'id' in edge.keys():
                        if edge['id'] not in tmpids:
                            tmpids.append(edge['id'])
                        else:
                            flag = False
                    if flag:
                        if 'uvPoints' in edge.keys() and 'uvNomals' in edge.keys():
                            pt_feat_list = edge['uvPoints']
                            norm_feat_list = edge['uvNomals']
                            pt_feat = torch.tensor(pt_feat_list).permute(1,0).to(device_info)
                            norm_feat = torch.tensor(norm_feat_list).permute(1,0).to(device_info)
                            pt_feat = pt_feat.view(3, args.sample_num, 1)
                            norm_feat = norm_feat.view(3, args.sample_num, 1)
                            feat = torch.cat([pt_feat, norm_feat], 0)
                            feat = feat.expand(6, args.sample_num, args.sample_num)
                        node_features.append(feat)

            # read vertices
            if 'vertices' in dict.keys():
                for vertex in dict['vertices']:
                    flag = True
                    if 'id' in vertex.keys():
                        if vertex['id'] not in tmpids:
                            tmpids.append(vertex['id'])
                        else:
                            flag = False
                    if flag:
                        if 'uvPoints' in vertex.keys() and 'uvNomals' in vertex.keys():
                            pt_feat_list = vertex['uvPoints']
                            norm_feat_list = vertex['uvNomals']

                            pt_feat = torch.tensor(pt_feat_list).to(device_info)
                            norm_feat = torch.tensor(norm_feat_list).to(device_info)
                            pt_feat = pt_feat.view(3, 1, 1)
                            norm_feat = norm_feat.view(3, 1, 1)
                            feat = torch.cat([pt_feat, norm_feat], 0)
                            feat = feat.expand(6, args.sample_num, args.sample_num)
                        node_features.append(feat)
            features.extend(node_features)

    return features


class MyDataSet:
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]
