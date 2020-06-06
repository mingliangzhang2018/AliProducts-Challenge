import os
from network import Network
from configs.configs_AiProducts_resnest50 import parse_args
from dataloader.AiProducts import AiProducts
from dataloader.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from torch.utils.data import DataLoader
from utils import *
from function import train_model, valid_model
import torch
import csv
import json

cfg = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
cfg.INPUT_SIZE = (cfg.INPUT_SIZE, cfg.INPUT_SIZE)

test_set = eval(cfg.DATASET)("test", cfg)
testLoader = DataLoader(
    test_set,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=True,
)

model = Network(cfg)

criterion = eval(cfg.LOSS_TYPE)().cuda()

optimizer = get_optimizer(cfg, model)

scheduler = get_scheduler(cfg, optimizer)

if cfg.RESUME_MODEL != "":
    checkpoint = torch.load(cfg.RESUME_MODEL, map_location="cuda")
    model.load_model(cfg.RESUME_MODEL)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    best_result = checkpoint['best_result']
    print("resume model from", cfg.RESUME_MODEL)

model = torch.nn.DataParallel(model).cuda()

def test_model(testLoader, model,cfg):
    model.eval()
    start_time = time.time()
    headers = ['id','predicted']
    rows = []
    #m = torch.nn.Softmax(dim=1)
    #annos = []
    #thresh = 0.95
    #save_path = "./test_choose.json"
    #dataroot = "/home2/zml/data/AliProduct/test/"
    with torch.no_grad():
        for batch_idx, (data, image_name) in enumerate(testLoader):
            # 针对Cuda进行设置
            data= data.cuda()
            # 模型前向传播
            output = model(data)
            #output = m(output)
            # 记录误差及准确率
            pred = output.data.max(1, keepdim=True)[1]  # 获得得分最高的类别
            for i in range(len(image_name)):
                rows.append((image_name[i],pred[i].data.item()))
                # if output[i][pred[i].data.item()].data.item() > thresh:
                    # annos.append({"category_id": pred[i].data.item(), "fpath": dataroot+image_name[i]})
    
    # with open(save_path, "w") as f:
        # json.dump({"annotations": annos, "num_classes": 50030}, f)
                    
    end_time = time.time()
    print('time consume is %.2f min' % ((end_time - start_time) / 60))
    if not os.path.exists("./output"):
        os.makedirs("./output")
    csv_name = "output/"+cfg.BACKBONE+"_"+cfg.RESUME_MODEL.split('/')[-1][:-4]+".csv"
    with open(csv_name,'w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

test_model(testLoader, model, cfg)


