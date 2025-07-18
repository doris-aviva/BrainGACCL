import json
import os
import time

import numpy as np
import torch

import random
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam

from option import parse
from process_data import get_dataset
from test_model import Evaluate
from train_model import finetune_model

from model import Net

from classifier import Classifier
from torch_geometric.data import DataLoader

from functions import evaluate


argv = parse()


def run(train_loader,
              test_loader,
              model,
              model_optimizer,
              classifier,
              classifier_optimizer,
              lr_decay_factor,
              lr_decay_factor_m,
              lr_decay_step_size,
              lr_decay_step_size_m,
              device,
              epochs):

    Train_loss_k = []
    train_acc = 0
    test_acc = 0

    train_accs, test_accs = [], []

    for epoch in range(1, epochs + 1):
        cls_loss = finetune_model(model, model_optimizer, classifier, classifier_optimizer, train_loader, device)

        print("epoch:{}, class_loss:{}".format(epoch, cls_loss))

        loss_accumulate, Pred, Prob, Label = Evaluate(model, classifier, train_loader, device, True)
        Train_loss_k.append(loss_accumulate)

        Prob = np.array(Prob)
        evaluation_metrics = evaluate(pred=Pred, prob=Prob, label=Label)
        print({'acc': evaluation_metrics[0],
               'sen': evaluation_metrics[1],
               'spe': evaluation_metrics[2],
               'pre': evaluation_metrics[3],
               'f1': evaluation_metrics[4],
               'auc': evaluation_metrics[5],
               'bac': evaluation_metrics[6],
               'loss': loss_accumulate})

        loss_accumulate, Pred, Prob, Label = Evaluate(model, classifier, test_loader, device, True)

        Prob = np.array(Prob)
        evaluation_metrics = evaluate(pred=Pred, prob=Prob, label=Label)
        print({'acc': evaluation_metrics[0],
               'sen': evaluation_metrics[1],
               'spe': evaluation_metrics[2],
               'pre': evaluation_metrics[3],
               'f1': evaluation_metrics[4],
               'auc': evaluation_metrics[5],
               'bac': evaluation_metrics[6]})

        # model_optimizer
        if epoch % lr_decay_step_size_m == 0:
            for param_group in model_optimizer.param_groups:
                param_group['lr'] = lr_decay_factor_m * param_group['lr']
        # classifier_optimizer
        if epoch % lr_decay_step_size == 0:
            for param_group in classifier_optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if argv.seed is not None:
        random.seed(argv.seed)
        np.random.seed(argv.seed)
        torch.manual_seed(argv.seed)
        torch.cuda.manual_seed(argv.seed)
        torch.cuda.manual_seed_all(argv.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset, labels = get_dataset()

    Time = []
    ACC_k, SEN_k, SPE_k, PRE_k, F1_k, AUC_k, BAC_k = [], [], [], [], [], [], []

    # Kfold
    num_folds = argv.k_fold
    train_ls_sum = 0
    val_ls_sum = 0

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=argv.random_state)

    fold = 1
    for train_index, test_index in skf.split(dataset, labels):
        test_dataset = [dataset[int(i)] for i in list(test_index)]
        test_loader = DataLoader(test_dataset, batch_size=argv.minibatch_size, shuffle=False)

        train_dataset = [dataset[int(i)] for i in list(train_index)]
        train_loader = DataLoader(train_dataset, batch_size=argv.minibatch_size, shuffle=True)

        Train_loss_k = []

        # model
        model = Net(argv.input_dim, argv.num_classes, argv.model_drop)

        model_optimizer = torch.optim.SGD(model.parameters(), lr=argv.lr_model, momentum=argv.momentum_m, weight_decay=argv.weight_decay_m)
        classifier = Classifier(argv.hidden_dim_c, argv.num_classes, argv.class_drop)
        classifier_optimizer = torch.optim.SGD(classifier.parameters(), lr=argv.lr, momentum=argv.momentum, weight_decay=argv.weight_decay)

        model.to(device)
        classifier.to(device)

        if argv.pretrained:
            if os.path.isfile(argv.pretrained):
                print("=> loading Wcheckpoint '{}'".format(argv.pretrained))
                checkpoint = torch.load(argv.pretrained, map_location="cpu")

                state_dict_model = checkpoint['model']

                msg = model.load_state_dict(state_dict_model, strict=False)

                print("=> loaded pre-trained model '{}'".format(argv.pretrained))
            else:
                print("=> no checkpoint found at '{}'".format(argv.pretrained))

        # train
        epochs = argv.num_epochs
        lr_decay_factor = argv.lr_decay_factor
        lr_decay_factor_m = argv.lr_decay_factor_m
        lr_decay_step_size = argv.lr_decay_step_size
        lr_decay_step_size_m = argv.lr_decay_step_size_m

        # start training
        start_time = time.time()
        run(train_loader,
                  test_loader,
                  model,
                  model_optimizer,
                  classifier,
                  classifier_optimizer,
                  lr_decay_factor,
                  lr_decay_factor_m,
                  lr_decay_step_size,
                  lr_decay_step_size_m,
                  device,
                  epochs)
        end_time = time.time()
        Time.append(end_time - start_time)

        Label = []
        Pred = []
        Prob = []
        Fea = []
        loss_accumulate, Pred, Prob, Fea, Label = Evaluate(model, classifier, test_loader, device, False)

        Prob = np.array(Prob)
        evaluation_metrics = evaluate(pred=Pred, prob=Prob, label=Label)
        print('\n', 'fold:', fold, 'result:')
        print({'acc': evaluation_metrics[0],
               'sen': evaluation_metrics[1],
               'spe': evaluation_metrics[2],
               'pre': evaluation_metrics[3],
               'f1': evaluation_metrics[4],
               'auc': evaluation_metrics[5],
               'bac': evaluation_metrics[6]})
        print('\n')

        ACC_k.append(evaluation_metrics[0])
        SEN_k.append(evaluation_metrics[1])
        SPE_k.append(evaluation_metrics[2])
        PRE_k.append(evaluation_metrics[3])
        F1_k.append(evaluation_metrics[4])
        AUC_k.append(evaluation_metrics[5])
        BAC_k.append(evaluation_metrics[6])

        fold += 1

    print('Total training time (s):', np.sum(Time))
    print('Average training time per fold (s):', np.mean(Time))

    mean_acc = np.mean(ACC_k)
    mean_sen = np.mean(SEN_k)
    mean_spe = np.mean(SPE_k)
    mean_pre = np.mean(PRE_k)
    mean_f1 = np.mean(F1_k)
    mean_auc = np.mean(AUC_k)
    mean_bac = np.mean(BAC_k)
    print('final result:')
    print({'acc': mean_acc, 'sen': mean_sen, 'spe': mean_spe, 'pre': mean_pre, 'f1': mean_f1, 'auc': mean_auc,
           'bac': mean_bac})

