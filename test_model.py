import torch
from torch import nn


def evaluate(model, classifier, data, device):
    model.eval()
    classifier.eval()

    with torch.no_grad():
        data = data.to(device)
        model = model.to(device)
        classifier = classifier.to(device)

        output_tensor = model.forward_cl(data.x, data.edge_index, data.edge_weight, data.batch, device)
        feature = output_tensor
        logit_score = classifier(output_tensor, device)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logit_score, data.y.long())

    return feature, logit_score, loss


def Evaluate(model, classifier, loader, device, isTrain):
    loss_accumulate = 0.0
    Label = []
    Pred = []
    Prob = []
    Fea = []

    for data in loader:
        data = data.to(device)
        fea, logit_score, loss = evaluate(model, classifier, data, device)

        Label.extend(data.y.tolist())
        pred = logit_score.argmax(1).tolist()
        Pred.extend(pred)
        prob = logit_score.softmax(1).tolist()
        Prob.extend(prob)
        Fea.extend(fea.tolist())
        loss_accumulate += loss.detach().cpu().numpy()

    if isTrain:
        return loss_accumulate, Pred, Prob, Label
    else:
        return loss_accumulate, Pred, Prob, Fea, Label
