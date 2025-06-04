import torch
import random
import torch.nn.functional as F


def loss_add_negative(x1, x2, x):
    T = 0.5
    batch_size, _ = x1.size()

    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix_a = torch.einsum('ik,jk->ij', x1, x2) / (torch.einsum('i,j->ij', x1_abs, x2_abs) + 1e-8)
    sim_matrix_a = torch.exp(sim_matrix_a / T)

    pos_sim_a = sim_matrix_a[range(batch_size), range(batch_size)]

    interpolated_negatives = []
    for k in range(batch_size):
        exclude_k = torch.arange(batch_size) != k
        random_index = torch.multinomial(exclude_k.float(), 1).item()
        neg_sample = 0.5 * x[k] + 0.5 * x[random_index]
        interpolated_negatives.append(neg_sample)

    interpolated_negatives = torch.stack(interpolated_negatives)

    interpolated_negatives_abs = interpolated_negatives.norm(dim=1)

    neg_sim_matrix_a = torch.einsum('ik,jk->ij', x1, interpolated_negatives) / (
        torch.einsum('i,j->ij', x1_abs, interpolated_negatives_abs) + 1e-8
    )
    neg_sim_matrix_a = torch.exp(neg_sim_matrix_a / T)

    mask_a = ~torch.eye(batch_size, dtype=torch.bool, device=x1.device)
    neg_sim_matrix_a = neg_sim_matrix_a.masked_select(mask_a).view(batch_size, -1)

    loss_a = pos_sim_a / (sim_matrix_a.sum(dim=1) - pos_sim_a + neg_sim_matrix_a.sum(dim=1) + 1e-8)
    loss_a = - torch.log(loss_a + 1e-8).mean()

    sim_matrix_b = torch.einsum('ik,jk->ij', x2, x1) / (torch.einsum('i,j->ij', x2_abs, x1_abs) + 1e-8)
    sim_matrix_b = torch.exp(sim_matrix_b / T)

    pos_sim_b = sim_matrix_b[range(batch_size), range(batch_size)]

    neg_sim_matrix_b = torch.einsum('ik,jk->ij', x2, interpolated_negatives) / (
            torch.einsum('i,j->ij', x2_abs, interpolated_negatives_abs) + 1e-8
    )
    neg_sim_matrix_b = torch.exp(neg_sim_matrix_b / T)

    mask_b = ~torch.eye(batch_size, dtype=torch.bool, device=x2.device)
    neg_sim_matrix_b = neg_sim_matrix_b.masked_select(mask_b).view(batch_size, -1)

    loss_b = pos_sim_b / (sim_matrix_b.sum(dim=1) - pos_sim_b + neg_sim_matrix_b.sum(dim=1) + 1e-8)
    loss_b = - torch.log(loss_b + 1e-8).mean()

    loss = (loss_a + loss_b) / 2
    return loss


def train_model_cl(attr1, attr2, stru1, stru2,
                   model, model_optimizer,
                   loader, device):

    model.train()
    attr1.eval()
    attr2.eval()
    stru1.eval()
    stru2.eval()

    loss_all = 0
    total_graphs = 0

    for data in loader:
        model_optimizer.zero_grad()

        data = data.to(device)
        _, view1, _ = attr1(data, True)
        _, view2, _ = attr2(data, True)

        _, view3 = stru1(data, device)
        _, view4 = stru2(data, device)

        input_list1 = [view1, view2]
        input_list2 = [view3, view4]
        view12 = random.choice(input_list1)
        view34 = random.choice(input_list2)

        output = model.forward_cl(data.x, data.edge_index, data.edge_weight, data.batch, device)
        output1 = model.forward_cl(view1.x, view1.edge_index, view1.edge_weight, view1.batch, device)
        output2 = model.forward_cl(view2.x, view2.edge_index, view2.edge_weight, view2.batch, device)
        output3 = model.forward_cl(view3.x, view3.edge_index, view3.edge_weight, view3.batch, device)
        output4 = model.forward_cl(view4.x, view4.edge_index, view4.edge_weight, view4.batch, device)
        output12 = model.forward_cl(view12.x, view12.edge_index, view12.edge_weight, view12.batch, device)
        output34 = model.forward_cl(view34.x, view34.edge_index, view34.edge_weight, view34.batch, device)

        loss_cl0 = loss_add_negative(output1, output2, output)
        loss_cl2 = loss_add_negative(output3, output4, output)
        loss_cl3 = loss_add_negative(output12, output34, output)

        cl_loss = (loss_cl0 + loss_cl2 + loss_cl3)/3

        loss = cl_loss
        loss_all += loss.item() * data.num_graphs
        cl_loss.backward()
        model_optimizer.step()

        total_graphs += data.num_graphs

    loss_all /= total_graphs

    return loss_all


def finetune_model(model, model_optimizer, classifier, classifier_optimizer, loader, device):
    model.train()
    classifier.train()
    loss_all = 0
    total_graphs = 0

    for data in loader:
        model_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        data = data.to(device)
        classifier = classifier.to(device)

        output_tensor = model.forward_cl(data.x, data.edge_index, data.edge_weight, data.batch, device)
        output = classifier(output_tensor, device)
        loss = F.nll_loss(output, data.y)

        loss_all += loss.item() * data.num_graphs

        loss.backward()
        model_optimizer.step()
        classifier_optimizer.step()

        total_graphs += data.num_graphs

    loss_all /= total_graphs
    return loss_all
