import os
from network import Network
from configs_AiProducts_resnest50 import parse_args
from dataloader.AiProducts import AiProducts
from dataloader.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from torch.utils.data import DataLoader
from utils import *
from function import train_model, valid_model
import torch

cfg = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
cfg.INPUT_SIZE = (cfg.INPUT_SIZE, cfg.INPUT_SIZE)

valid_set = eval(cfg.DATASET)("valid", cfg)
validLoader = DataLoader(
    valid_set,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=True,
)

model = Network(cfg)

criterion = eval(cfg.LOSS_TYPE)().cuda()

optimizer = get_optimizer(cfg, model)

scheduler = get_scheduler(cfg, optimizer)

log_dir = os.path.join("./log_tau", cfg.DATASET)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if cfg.RESUME_MODEL != "":
    checkpoint = torch.load(cfg.RESUME_MODEL, map_location="cuda")
    model.load_model(cfg.RESUME_MODEL)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    best_result = checkpoint['best_result']
    print("resume model from", cfg.RESUME_MODEL)

model = torch.nn.DataParallel(model).cuda()

model_state_dict = model.state_dict()
# print(model_state_dict.keys())

# set bias as zero
model_state_dict['module.classifier.bias'].copy_(torch.zeros(
    (cfg.num_classes)))
weight_ori = model_state_dict['module.classifier.weight']
norm_weight = torch.norm(weight_ori, 2, 1)
best_result = 0
pt = 0

for p in np.linspace(0, 2, 10):

    ws = weight_ori.clone()
    for i in range(weight_ori.size(0)):
        ws[i] = ws[i] / torch.pow(norm_weight[i], p)

    model_state_dict['module.classifier.weight'].copy_(ws)

    valid_acc, valid_loss = valid_model(validLoader, model, 0, criterion, cfg)

    if valid_acc > best_result:
        best_result = valid_acc
        pt = p
        torch.save(
            {
                'state_dict': model.state_dict(),
                'epoch': 0,
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_result': best_result,
            }, os.path.join(log_dir, "best_model.pth"))
    print("when p is %0.2f, valid_acc is %0.2f" % (p, valid_acc))
    print("__________________________________________________________________")

print("best result is %0.2f when p is %0.2f" % (best_result, pt))

