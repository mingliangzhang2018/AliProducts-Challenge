import os
from network import Network
from configs.configs_AiProducts_resnest50 import parse_args
from dataloader.AiProducts import AiProducts
from dataloader.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import *
from function import train_model, valid_model

cfg = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
cfg.INPUT_SIZE = (cfg.INPUT_SIZE, cfg.INPUT_SIZE)
cudnn.benchmark = True

train_set = eval(cfg.DATASET)("train", cfg)
trainLoader = DataLoader(train_set,
                         batch_size=cfg.BATCH_SIZE,
                         shuffle=True,
                         num_workers=cfg.NUM_WORKERS,
                         pin_memory=True)

valid_set = eval(cfg.DATASET)("valid", cfg)
validLoader = DataLoader(
    valid_set,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=True,
)

model = Network(cfg)
if cfg.BACKBONE_FREEZE:
    model.freeze_backbone()
    print("Backbone has been freezed!")
    
model = torch.nn.DataParallel(model).cuda()

criterion = eval(cfg.LOSS_TYPE)().cuda()

optimizer = get_optimizer(cfg, model)

scheduler = get_scheduler(cfg, optimizer)

log_dir = os.path.join("./output", cfg.DATASET)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

start_epoch, best_result, best_epoch = 0, 0, 0

if cfg.RESUME_MODEL != "":
    checkpoint = torch.load(cfg.RESUME_MODEL, map_location="cuda")
    # model.load_model(cfg.RESUME_MODEL)
    model.load_state_dict(checkpoint['state_dict'])
    if not cfg.BACKBONE_FREEZE:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
    print("resume model from", cfg.RESUME_MODEL)

print("method of sampler is", cfg.SAMPLER_TYPE)



for epoch in range(start_epoch, cfg.MAX_EPOCH):

    scheduler.step()
    if epoch == 50:
        params=[]
        for name, p in model.named_parameters():
            if p.requires_grad:
                params.append({"params": p})
        if cfg.OPTIMIZER == "SGD":
            optimizer = torch.optim.SGD(params, lr=1e-4, momentum=0.9, weight_decay=cfg.WEIGHT_DECAY)

    if epoch==cfg.CHANGE_SAMPLER_EPOCH:
        cfg.SAMPLER_TYPE = "class_balance"
        train_set = eval(cfg.DATASET)("train", cfg)
        trainLoader = DataLoader(train_set,
                         batch_size=cfg.BATCH_SIZE,
                         shuffle=True,
                         num_workers=cfg.NUM_WORKERS,
                         pin_memory=True)

    lr = next(iter(optimizer.param_groups))['lr']

    print("Learning rate is",lr, ", Sampler method is", cfg.SAMPLER_TYPE)

    train_acc, train_loss = train_model(trainLoader, model, epoch, optimizer,
                                        criterion, cfg)

    if epoch % cfg.SAVE_STEP == 0:
        model_save_path = os.path.join(log_dir, "epoch_{}.pth".format(epoch))
        torch.save(
            {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_result': best_result
            }, model_save_path)

    valid_acc, valid_loss = valid_model(validLoader, model, epoch, criterion,
                                        cfg)

    if valid_acc > best_result:
        best_result = valid_acc
        best_epoch = epoch
        torch.save(
            {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_result': best_result,
            }, os.path.join(log_dir, "best_model.pth"))
    print("\n________________________best epoch is %d, best results is %.2f______________________\n" \
              %(best_epoch,best_result))
