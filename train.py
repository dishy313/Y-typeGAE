
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import time
import args
import modelGATE as GATE
from torch.utils import tensorboard as tb
from dataset_UVBrepWeight import *
import torch.utils.data as Data

outf = 'out\\GATE_'  # save path
curTimeStr = time.strftime('%y%m%d%H%M', time.localtime(time.time()))
outf = outf + curTimeStr
save_model_path = 'model\\model_' +  curTimeStr

curLogPath = 'logs\\' + curTimeStr  # log path

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

def get_feature_acc(feat_rec, feat_label):
    feat_sub = feat_label - feat_rec
    feat_mean = torch.mean(feat_label, dim=1)
    feat_mean.repeat(1, args.default_node_dim, args.input_dim)
    feat_mean_sub = feat_label - feat_mean
    accuracy = 1 - torch.sum(torch.pow(feat_sub, 2)) / torch.sum(torch.pow(feat_mean_sub, 2))
    return accuracy

colormap = ['forestgreen', 'b', 'red', 'black']

if __name__ == '__main__':
    os.makedirs(outf, exist_ok=True)
    os.makedirs(curLogPath, exist_ok=True)

    #load CNNAE
    CNNAE_model_path = '.\\model_CNNAE\\model_CNNAE_Tag80_2411191535_epoch9778_acc0.9852436780929565.pth'
    model_CNNAE = torch.load(CNNAE_model_path, map_location=device)

    # init model and optimizer
    model = GATE.GATE(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    train_path = ".\\datas"
    validation_path = ".\\datas"

    max_structure_acc = 0.9
    max_feature_acc = 0.9
    max_acc = 0.9

    max_val_structure_acc = 0.6
    max_val_feature_acc = 0.7
    max_val_acc = 0.7

    featuresBrep, adjBrep = make_graph_data(train_path, model_CNNAE, device)

    loader = Data.DataLoader(MyDataSet(featuresBrep, adjBrep), args.batch_size, True)

    writer = tb.SummaryWriter(curLogPath)
    fid = open('{}/loss_and_accurary.txt'.format(outf), 'a')

    # train model
    for epoch in range(args.num_epoch):
        model.train()
        length_b = 0
        total_loss = 0
        total_train_acc = 0
        total_feature_loss = 0
        total_feature_acc = 0
        total_feature_part1_acc = 0
        total_feature_part2_acc = 0
        total_structure_loss = 0
        total_structure_acc = 0

        for features, adj_ in loader:
            length_b += 1
            feat_pred, adj_pred, H, Cout1, Cout2 = model(features, adj_)
            optimizer.zero_grad()
            features_loss = torch.sqrt(torch.sum(torch.pow(features - feat_pred, 2)))

            weight = torch.ones(adj_pred.shape).to(device)
            adj_ori_sum = torch.sum(adj_, dim=1)
            adj_ori_length = torch.sum((adj_ori_sum > 0).long())
            weight_val = torch.sum(adj_ori_sum) / adj_ori_length
            weight = torch.where(adj_ > 0, weight_val, weight).view(-1)
            structure_loss_l = F.cross_entropy(adj_pred.view(-1), adj_.view(-1), weight=weight)
            structure_loss = torch.mean(structure_loss_l)

            # Total loss
            loss = 0.5 * structure_loss + 0.5 * features_loss
            structure_acc = get_structure_acc(adj_pred, adj_)
            feature_acc = get_feature_acc(feat_pred, features)
            train_acc = 0.5 * structure_acc + 0.5 * feature_acc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()

            total_loss += loss.item()
            total_train_acc += train_acc
            total_feature_loss += features_loss.item()
            total_feature_acc += feature_acc
            total_structure_loss += structure_loss.item()
            total_structure_acc += structure_acc

        final_loss_val = total_loss / length_b
        final_acc_val = total_train_acc / length_b
        final_feature_loss_val = total_feature_loss / length_b
        final_feature_acc_val = total_feature_acc / length_b
        final_feature_part1_acc_val = total_feature_part1_acc / length_b
        final_feature_part2_acc_val = total_feature_part2_acc / length_b
        final_structure_loss_val = total_structure_loss / length_b
        final_structure_acc_val = total_structure_acc / length_b

        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(final_loss_val),
              "(", "{:.5f}".format(final_feature_loss_val), "+", "{:.5f}".format(final_structure_loss_val), ")",
              "train_acc=", "{:.5f}".format(final_acc_val),
              "(0.5*", "{:.5f}".format(final_feature_acc_val), "+0.5*", "{:.5f}".format(final_structure_acc_val), ")")
        fid.write('[%d/%d] train loss: %f  train acc: %f\n' % (epoch, args.num_epoch, final_loss_val,final_acc_val))
        fid.write('[%d/%d] feature loss: %f  feature acc: %f\n' % (epoch, args.num_epoch, final_feature_loss_val,final_feature_acc_val))
        fid.write('[%d/%d] structure loss: %f  structure acc: %f\n' % (epoch, args.num_epoch, final_structure_loss_val,final_structure_acc_val))
        writer.add_scalars("Loss",
                           {'train-loss': final_loss_val, 'feature-loss': final_feature_loss_val,
                            'structure-loss': final_structure_loss_val}, epoch)
        writer.add_scalars("Accuracy",
                           {'train-acc': final_acc_val, 'feature-acc': final_feature_acc_val,
                            'structure-acc': final_structure_acc_val}, epoch)
        if epoch > 10:
            if final_acc_val > max_acc:
                model_name_end = save_model_path + f'best_acc_{"{:.8f}".format(final_acc_val)}_epoch{epoch}.pth'
                torch.save(model, model_name_end)
                max_acc = final_acc_val

    model_name_end = save_model_path + f'_final.pth'
    torch.save(model, model_name_end)
    fid.close()