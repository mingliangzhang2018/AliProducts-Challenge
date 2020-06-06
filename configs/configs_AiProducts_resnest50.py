import argparse


def parse_args():
     parser = argparse.ArgumentParser(description="codes for classifier-balancing-test")
     parser.add_argument(
          "--gpu_ids",
          help="decide which gpu to use",
          default="0,1,2,3",
          type=str
     )
     parser.add_argument(
          "--DATASET",
          help="choose one dataset (cifar10, cifar100, AiProducts)",
          default="AiProducts",
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
          default="",
          type=str
     )
     parser.add_argument(
          "--BASE_LR",
          help="base learning rate of optimizer",
          default=0.5,
          type=float
     )
     parser.add_argument(
          "--WEIGHT_DECAY",
          help="weight decay of optimizer",
          default=1e-4,
          type=float
     )
     parser.add_argument(
          "--LR_SCHEDULER",
          help="type of learning scheduler (multistep, cosine)",
          default="multistep",
          type=str
     )

     parser.add_argument(
          "--BATCH_SIZE",
          help="batch size of dataset",
          default=512, 
          type=int
     )
     parser.add_argument(
          "--NUM_WORKERS",
          help="number workers of CPU",
          default=16, 
          type=int
     )
     parser.add_argument(
          "--MAX_EPOCH",
          help="epoch number of train",
          default=120, 
          type=int
     )

     parser.add_argument(
          "--SAMPLER_TYPE",
          help="method of sampler (instance_balance, class_balance, reverse)",
          default="instance_balance",
          type=str
     )
     parser.add_argument(
          "--INPUT_SIZE",
          help="the size of input image",
          default=112,
          type=int
     )
     parser.add_argument(
          "--COLOR_SPACE",
          help="color space of input image",
          default="BGR",
          type=str
     )
     parser.add_argument(
          "--DATASET_ROOT",
          help="dataset root",
          default="./dataset",
          type=str
     )
     parser.add_argument(
          "--DATASET_TRAIN_JSON",
          help="path of train dataset index",
          default="./dataset/AiProducts/converted_train.json",
          type=str
     )
     parser.add_argument(
          "--DATASET_VALID_JSON",
          help="path of valid dataset index",
          default="./dataset/AiProducts/converted_val.json",
          type=str
     )
     parser.add_argument(
          "--DATASET_TEST_JSON",
          help="path of test dataset index",
          default="./dataset/AiProducts/converted_test.json",
          type=str
     )
     parser.add_argument(
          "--PRETRAINED_BACKBONE",
          help="decide which PRETRAINED_BACKBONE to use",
          default="",
          type=str
     )
     parser.add_argument(
          "--SHOW_STEP",
          help="steps of loss displayment ",
          default=500, 
          type=int
     )
     parser.add_argument(
          "--SAVE_STEP",
          help="steps of save model ",
          default=5, 
          type=int
     )
     parser.add_argument(
          "--num_classes",
          help="number of classes",
          default=50030, 
          type=int
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
          "--CLASSIFIER",
          help="decide which CLASSIFIER to use FC or FCNorm",
          default="FC", 
          type=str
     )
     parser.add_argument(
          "--BACKBONE",
          help="decide which BACKBONE to use res50, res100,res32_cifar, resnest50",
          default="resnest50",
          type=str
     )
     parser.add_argument(
          "--DECAY_STEP",
          help="decay step",
          default=[10,20,30],
          type=list
     )
     parser.add_argument(
          "--CHANGE_SAMPLER_EPOCH",
          help="epoch when change the sampler mothod",
          default=40,
          type=int
     )
     parser.add_argument(
          "--p",
          help="p",
          default=0,
          type=float
     )

     args = parser.parse_args()
     return args
