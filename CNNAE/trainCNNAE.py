from torch import nn
from torch.optim import Adam
import time
import args
import modelCNNAE as CNNAE
from torch.utils import tensorboard as tb
from dataset_CNNAE import *
import torch.utils.data as Data

outf = 'out\\GATE_'  # save path
curTimeStr = time.strftime('%y%m%d%H%M', time.localtime(time.time()))
outf = outf + curTimeStr
save_model_path = 'model\\' + curTimeStr

curLogPath = 'logs\\' + curTimeStr  # tensorboard

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_feature_acc(feat_rec, feat_label):
    feat_sub = feat_label - feat_rec
    accuracy = 1 - torch.sqrt(torch.sum(torch.pow(feat_sub, 2))) / torch.sqrt(
        torch.sum(torch.pow(torch.ones(feat_rec.shape).to(device), 2)))
    return accuracy


if __name__ == '__main__':
    os.makedirs(outf, exist_ok=True)
    os.makedirs(curLogPath, exist_ok=True)

    max_acc = 0.9

    # init model and optimizer
    model = CNNAE.CNNAE()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    json_path = ".\\datas"  # json files of BRep model

    featuresBrep = make_graph_data(json_path, device, args.batch_size_CNNAE)

    loader = Data.DataLoader(MyDataSet(featuresBrep), args.batch_size_CNNAE, True)

    writer = tb.SummaryWriter(curLogPath)
    fid = open('{}/loss_and_accurary.txt'.format(outf), 'a')

    # train model
    for epoch in range(args.num_epoch_CNNAE):

        total_loss = 0
        total_train_acc = 0
        total_feature_loss = 0
        total_feature_acc = 0
        indexNum = 0

        for features in loader:
            feat_pred = model(features)

            optimizer.zero_grad()
            features_loss = torch.sqrt(torch.sum(torch.pow(features - feat_pred, 2)))

            # Total loss
            loss = features_loss

            feature_acc = get_feature_acc(feat_pred, features)
            train_acc = feature_acc

            loss.backward()

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)

            optimizer.step()

            total_loss += loss.item()
            total_train_acc += train_acc
            total_feature_loss += features_loss.item()
            total_feature_acc += feature_acc

            indexNum += 1

        final_loss_val = total_loss / indexNum
        final_acc_val = total_train_acc / indexNum
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(final_loss_val),
              "train_acc=", "{:.5f}".format(final_acc_val))
        fid.write(
            '[%d/%d] train loss: %f  train acc: %f\n' % (epoch, args.num_epoch_CNNAE, final_loss_val, final_acc_val))
        writer.add_scalars("Loss",
                           {'train-loss': final_loss_val}, epoch)
        writer.add_scalars("Accuracy",
                           {'train-acc': final_acc_val}, epoch)

        if final_acc_val > max_acc:
            model_name_end = save_model_path + f'_epoch{epoch}_acc{final_acc_val}.pth'
            torch.save(model, model_name_end)
            max_acc = final_acc_val

    model_name_end = save_model_path + f'_final.pth'
    torch.save(model, model_name_end)
    fid.close()
