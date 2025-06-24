import args
from dataset_UVBrepWeight import *
import torch.utils.data as Data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_structure_acc(adj_rec, adj_label):

    labels_all = adj_label.view(-1)
    preds_all = adj_rec + torch.ones(adj_rec.shape).cuda()*0.5
    preds_all = preds_all.type(torch.int64)
    A_pred_T = preds_all.permute(0, 2, 1)
    A_pred_U = torch.where(preds_all == A_pred_T, preds_all, 0)

    preds_all_tmp = A_pred_U.view(-1)
    accuracy = (preds_all_tmp == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def get_feature_accRMSE(feat_rec, feat_label):
    feat_sub = feat_label - feat_rec
    accuracy = torch.sqrt(torch.sum(torch.pow(feat_sub, 2))) / args.default_node_dim
    return accuracy

def get_feature_accR2(feat_rec, feat_label):
    feat_sub = feat_label - feat_rec
    feat_mean = torch.mean(feat_label, dim=1)
    feat_mean.repeat(1, args.default_node_dim, args.input_dim)
    feat_mean_sub = feat_label - feat_mean
    accuracy = 1 - torch.sum(torch.pow(feat_sub, 2)) / torch.sum(torch.pow(feat_mean_sub, 2))
    return accuracy

if __name__ == '__main__':

    #load CNNAE
    unet_model_name = '.\\model_CNNAE\\model_CNNAE_Tag80_2411191535_epoch9778_acc0.9852436780929565.pth'
    model_CNNAE = torch.load(unet_model_name, map_location=device)

    model_path = 'TrainedModel'

    model = torch.load(model_path, map_location=device)

    json_test_path = ".\\datas"

    featuresBrep_test, adjBrep_test = make_graph_data(json_test_path, model_CNNAE, device)

    loader = Data.DataLoader(MyDataSet(featuresBrep_test, adjBrep_test), 1, True)

    length_test = 0
    total_valid_acc = 0
    total_valid_feature_acc = 0
    total_valid_structure_acc = 0
    total_valid_feature_acc_RMSE = 0

    for features, adj_ in loader:
        length_test += 1
        feat_pred, adj_pred, H, Cout1, Cout2 = model(features, adj_)

        structure_acc = get_structure_acc(adj_pred, adj_)
        feature_acc_R2 = get_feature_accR2(feat_pred, features)
        feature_acc_RMSE = get_feature_accRMSE(feat_pred, features)

        train_acc = 0.5 * structure_acc + 0.5 * feature_acc_R2

        total_valid_acc += train_acc
        total_valid_feature_acc += feature_acc_R2
        total_valid_structure_acc += structure_acc
        total_valid_feature_acc_RMSE += feature_acc_RMSE

        print("OneAccR2 =", "{:.5f}".format(train_acc),
              "(0.5*", "{:.5f}".format(feature_acc_R2), "+0.5*", "{:.5f}".format(structure_acc),
              ")")
        print("OneFeatAccRMSE =", "{:.5f}".format(feature_acc_RMSE))

    final_acc_val = total_valid_acc / length_test
    final_feature_acc_val = total_valid_feature_acc / length_test
    final_feature_acc_val_RMSE = total_valid_feature_acc_RMSE / length_test
    final_structure_acc_val = total_valid_structure_acc / length_test


    print("Acc =", "{:.5f}".format(final_acc_val),
          "(0.5*", "{:.5f}".format(final_feature_acc_val), "+0.5*", "{:.5f}".format(final_structure_acc_val),
          ")")
    print("FeatAccRMSE =", "{:.5f}".format(final_feature_acc_val_RMSE))