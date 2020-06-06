import argparse


def parse_args():
     parser = argparse.ArgumentParser(description="codes for classifier-balancing-test")
     parser.add_argument(
          "--gpu_ids",
          help="decide which gpu to use",
          default="7",
          type=str
     )
     parser.add_argument(
          "--DATASET",
          help="choose one dataset (IMBALANCECIFAR10, IMBALANCECIFAR100, AiProducts)",
          default="IMBALANCECIFAR100",
          type=str
     )
     parser.add_argument(
          "--BACKBONE",
          help="decide which BACKBONE to use res50 or res32_cifar",
          default="res32_cifar",
          type=str
     )
     parser.add_argument(
          "--BACKBONE_FREEZE",
          help="backbone freeze and train the classifier",
          action="store_true",
          default=False
     )
     parser.add_argument(
          "--RESUME_MODEL",
          help="resume model",
          default="./pretrain_models/best_model.pth",
          type=str
     )
     parser.add_argument(
          "--BASE_LR",
          help="base learning rate of optimizer",
          default=0.1,
          type=float
     )
     parser.add_argument(
          "--WEIGHT_DECAY",
          help="weight decay of optimizer",
          default=2e-4,
          type=float
     )
     parser.add_argument(
          "--LR_SCHEDULER",
          help="type of learning scheduler (multistep, cosine)",
          default="cosine",
          type=str
     )
     parser.add_argument(
          "--num_classes",
          help="number of classes",
          default=100, 
          type=int
     )
     parser.add_argument(
          "--MAX_EPOCH",
          help="epoch number of train",
          default=40, 
          type=int
     )
     parser.add_argument(
          "--IMBALANCECIFAR_RATIO",
          help="imbalance ratio cifar dataset ",
          default=0.01,
          type=float
     )
     parser.add_argument(
          "--SAMPLER_TYPE",
          help="method of sampler (instance_balance, class_balance, reverse)",
          default="reverse",
          type=str
     )
     parser.add_argument(
          "--INPUT_SIZE",
          help="the size of input image",
          default=32,
          type=int
     )
     parser.add_argument(
          "--DATASET_ROOT",
          help="dataset root",
          default="",
          type=str
     )
     parser.add_argument(
          "--IMBALANCECIFAR_RANDOM_SEED",
          help="random seed for cifar dataset ",
          default=0,
          type=int
     )
     parser.add_argument(
          "--SHOW_STEP",
          help="steps of loss displayment ",
          default=10, 
          type=int
     )
     parser.add_argument(
          "--SAVE_STEP",
          help="steps of save model ",
          default=5, 
          type=int
     )
     parser.add_argument(
          "--NUM_WORKERS",
          help="number workers of CPU",
          default=8, 
          type=int
     )
     parser.add_argument(
          "--CLASSIFIER",
          help="decide which CLASSIFIER to use FC or FCNorm",
          default="FC", 
          type=str
     )
     parser.add_argument(
          "--PRETRAINED_BACKBONE",
          help="decide which PRETRAINED_BACKBONE to use",
          default="",
          type=str
     )
     parser.add_argument(
          "--LOSS_TYPE",
          help="loss type",
          default="CrossEntropy",
          type=str
     )
     parser.add_argument(
          "--OPTIMIZER",
          help="type of optimizer (SGD, ADAM)",
          default="SGD",
          type=str
     )
     parser.add_argument(
          "--BATCH_SIZE",
          help="batch size of dataset",
          default=256, 
          type=int
     )
     args = parser.parse_args()
     return args
