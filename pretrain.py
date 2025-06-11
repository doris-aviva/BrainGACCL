import shutil
import random
import torch
from torch.optim import Adam
import torch.backends.cudnn as cudnn

from option import parse

from train_model import train_model_cl
from train_generator import train_view_with_cls, train_cls
from Encoders import SAencoder, AAencoder
from structure_augmentor import StructureAugmentor
from process_data import get_dataset
from model import Net
from classifier import Classifier
from attribute_augmentor import AttributeAugmentor
from torch_geometric.data import DataLoader

from sklearn.model_selection import train_test_split


argv = parse()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def run(train_loader,
         val_loader, model, attr1, attr2,
              model_optimizer, attr_optimizer, classifier, classifier_optimizer,
              stru1, stru2, stru_optimizer, lr_decay_factor,
              lr_decay_step_size, device, epochs):

    model = model.to(device)
    attr1 = attr1.to(device)
    attr2 = attr2.to(device)
    stru1 = stru1.to(device)
    stru2 = stru2.to(device)
    classifier = classifier.to(device)

    for epoch in range(1, epochs + 1):
        model_loss = train_model_cl(attr1, attr2, stru1, stru2, model,
                                        model_optimizer, train_loader, device)
        attr_loss, stru_loss = train_view_with_cls(attr1, attr2, attr_optimizer,
                                                      stru1, stru2,
                                                      stru_optimizer, model, classifier,
                                                      val_loader, device)
        cls_loss = train_cls(model, classifier,
                             classifier_optimizer, val_loader, device)

        print("epoch:{}, model_loss:{}, attr_loss:{}, stru_loss:{}, class_loss:{}".format(epoch, model_loss,
                                                                                                      attr_loss,
                                                                                                      stru_loss,
                                                                                                      cls_loss))
        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
            }, is_best=False, filename=argv.filename.format(epoch))

        # attr_optimizer
        if epoch % lr_decay_step_size == 0:
            for param_group in attr_optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']
        # stru_optimizer
        if epoch % lr_decay_step_size == 0:
            for param_group in stru_optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']
        # model_optimizer
        if epoch % lr_decay_step_size == 0:
            for param_group in model_optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']
        # classifier_optimizer
        if epoch % lr_decay_step_size == 0:
            for param_group in classifier_optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if argv.seed is not None:
        random.seed(argv.seed)
        torch.manual_seed(argv.seed)
        cudnn.deterministic = True

    dataset, labels = get_dataset()

    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=argv.minibatch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=argv.minibatch_size, shuffle=True)
    
    # model
    model = Net(argv.input_dim, argv.num_classes, argv.model_drop)
    attr1 = AttributeAugmentor(argv.input_dim, argv.hidden_dim_attr_aug, AAencoder)
    attr2 = AttributeAugmentor(argv.input_dim, argv.hidden_dim_attr_aug, AAencoder)
    model_optimizer = Adam(model.parameters(), lr=argv.lr_model, weight_decay=argv.weight_decay_m)
    attr_optimizer = Adam([{'params': attr1.parameters()},
                           {'params': attr2.parameters()}], lr=argv.lr_aug
                          , weight_decay=argv.weight_decay_aug)

    stru1 = StructureAugmentor(
        SAencoder(num_dataset_features=argv.input_dim, emb_dim=argv.hidden_dim_stru_aug, num_gc_layers=argv.num_layers,
                  s_drop=argv.s_drop),
        mlp_dim=argv.mlp_dim).to(device)
    stru2 = StructureAugmentor(
        SAencoder(num_dataset_features=argv.input_dim, emb_dim=argv.hidden_dim_stru_aug, num_gc_layers=argv.num_layers,
                  s_drop=argv.s_drop),
        mlp_dim=argv.mlp_dim).to(device)

    stru_optimizer = torch.optim.Adam([{'params': stru1.parameters()},
                                               {'params': stru2.parameters()}], lr=argv.lr_aug, weight_decay=argv.weight_decay_aug)

    classifier = Classifier(argv.hidden_dim_c, argv.num_classes, argv.class_drop)
    classifier_optimizer = Adam(classifier.parameters(), lr=argv.lr, weight_decay=argv.weight_decay)

    batch_size = argv.minibatch_size
    epochs = argv.num_epochs
    lr_decay_factor = argv.lr_decay_factor
    lr_decay_step_size = argv.lr_decay_step_size
    
    run(train_loader,
         val_loader, model, attr1, attr2,
              model_optimizer, attr_optimizer, classifier, classifier_optimizer,
              stru1, stru2, stru_optimizer, lr_decay_factor,
              lr_decay_step_size, device, epochs)
        