import os
import csv
import time
import torch
import shutil
import random
import string
from tqdm import tqdm
from PIL import Image
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import SGD, Adam, AdamW
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from .loss import calc_loss
np.seterr(divide='ignore',invalid='ignore')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_result_dict(evaluator, epoch, start_since, end_since, loss, num):
    accuracy = evaluator.OverallAccuracy()
    presion = evaluator.Precision() # 一个列表包含每个类的查准率
    recall = evaluator.Recall() # 一个列表包含每个类的查全率
    F1score = evaluator.F1Score()  # 一个列表包含每个类的F1值
    Iou = evaluator.IntersectionOverUnion() # 一个列表包含每个类的Iou值
    FWIou = evaluator.Frequency_Weighted_Intersection_over_Union() 
    mIou = evaluator.MeanIntersectionOverUnion()
    result_dict = {'epoch':epoch+ 1,                           'accuracy': accuracy,    
                   'presion': (presion[0] + presion[1])/2.0,   'recall': (recall[0] + recall[1])/2.0,    'F1score': (F1score[0] + F1score[1])/2.0, 
                   'FWIou': FWIou,            'mIou': mIou,              'loss': loss/num,               'duration':time_trans(start_since, end_since),
                   'recall 0': recall[0],     'recall 1': recall[1],     'F1score 0': F1score[0],        'F1score 1': F1score[1],      'Iou 0': Iou[0],         'Iou 1': Iou[1],       
                   'presion 0': presion[0],   'presion 1': presion[1]}
    return  result_dict


def save_dict2csv(result_dict, csv_path):
    # 将字典的键作为表头名
    fieldnames = result_dict.keys()
    try:
        # 以追加模式打开文件，并写入字典内容
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            # 检查文件是否为空文件
            if file.tell() == 0:
                # 写入表头名
                writer.writeheader()
            # 写入字典内容
            writer.writerow(result_dict)
    except PermissionError:
            with open(csv_path[:-4] + '_temp.csv', mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                # 检查文件是否为空文件
                if file.tell() == 0:
                    # 写入表头名
                    writer.writeheader()
                # 写入字典内容
                writer.writerow(result_dict)

def time_trans(start, end):
    duration_in_seconds = end - start
    # 将时长转换为分钟和秒
    hours = int(duration_in_seconds // 3600)
    minutes = int((duration_in_seconds % 3600) // 60)
    seconds = int(duration_in_seconds % 60)
    if hours == 0:
        return f'{minutes} m {seconds} s'
    else:
        return f"{hours} h {minutes} m {seconds} s"





class Model_save_log(object):
    def __init__(self, save_path, max_epoch):
        self.nickname = os.path.basename(save_path)
        self.save_path = save_path
        self.max_epoch = max_epoch

        self.train_csv = self.save_path + os.sep + self.nickname + '_train_results.csv'
        self.val_csv   = self.save_path + os.sep + self.nickname + '_val_results.csv'
        self.best_weight = ''
        # self.check_csv(self.train_csv, self.val_csv)
        self.check_csv()
        self.train_losses = []
        self.train_accuracyes = []
        self.test_losses = []
        self.test_accuracyes  = []

        self.best_accuaryes = 0.0
        self.best_iou_1     = 0.0
        self.best_F1        = 0.0
        self.best_val_loss  = 1000000.0
        print(f'权重保存策略：保存在 测试精度、IOU_1，损失方面提高的模型，保存地址为： {self.save_path}')
    def get_val_result(self, train_result, val_result, model_dict, optimizer_dict, epoch):
        # 先保存权重
        if val_result['Iou 1'] > self.best_iou_1:
        # if val_result['accuracy'] > self.best_accuaryes or val_result['Iou 1'] > self.best_iou_1 or val_result['loss'] < self.best_val_loss or val_result['F1score'] > self.best_F1:
            self.best_val_loss  = val_result['loss']
            self.best_accuaryes = val_result['accuracy']
            self.best_iou_1     = val_result['Iou 1'] 
            self.best_F1        = val_result['F1score']
            if os.path.exists(self.best_weight):
                os.remove(self.best_weight)
            torch.save({
            'epoch': epoch + 1,
            'state_dict': model_dict,
            'optimizer' : optimizer_dict,
            }, self.save_path + os.sep + self.nickname + f'_{epoch+1}_{self.max_epoch}.pth.tar')

            self.best_weight = self.save_path + os.sep + self.nickname + f'_{epoch+1}_{self.max_epoch}.pth.tar'
        # 记录损失与精度
        self.train_losses.append(train_result['loss'])
        self.train_accuracyes.append(train_result['accuracy'])

        self.test_losses.append(val_result['loss'])
        self.test_accuracyes.append(val_result['accuracy'])

        save_dict2csv(train_result,  self.train_csv)
        save_dict2csv(val_result,    self.val_csv)

    ### 可视化训练过程的loss和accuracy
    def plot_fig(self):
        plt.style.use("ggplot")
        plt.figure(figsize=(10,6))
        plt.plot(np.arange(1, 1 + self.max_epoch), np.array(self.train_losses), label="train_loss")
        plt.plot(np.arange(1, 1 + self.max_epoch), np.array(self.test_losses), label="val_loss")
        plt.plot(np.arange(1, 1 + self.max_epoch), np.array(self.train_accuracyes), label="train_accuracyes")
        plt.plot(np.arange(1, 1 + self.max_epoch), np.array(self.test_accuracyes), label="valid_accuracyes")
        plt.ylim(0, 1)
        plt.title("Training and Validation Loss / Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss / Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(self.save_path + os.sep +  self.nickname + '_Loss_Accuracy_epoch.png',dpi = 600)
    def check_csv(self):
        # andom_string = '_' + ''.join(random.choices(string.ascii_letters + string.digits, k=5)) + '.csv'
        # if os.path.exists(self.train_csv):
        #     self.train_csv = self.train_csv[:-4] + andom_string
        # if os.path.exists(self.val_csv):
        #     self.val_csv = self.val_csv[:-4] + andom_string
        if os.path.exists(self.train_csv):
            os.remove(self.train_csv)
        if os.path.exists(self.val_csv):
            os.remove(self.val_csv)
class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v




def training(train_loader, model,  optimizer, epoch, evaluator, args):
    train_loss = 0.0
    total_num = 0.0
    start_since = time.time()
    device = args.device
    model.train()
    evaluator.reset()
    tbar = tqdm(train_loader, desc='Training>>>>>>>', leave=False)
    for i, batch in enumerate(tbar):
        x = batch[0]
        y = batch[1]

        rgb, target= x.to(device), y.to(device)
        rgb = F.interpolate(rgb, scale_factor=4, mode='bilinear')
        output = model(rgb.float())

        loss = calc_loss(output, target.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_num +=  x.size(0)
        train_loss += loss.data.cpu().numpy() * x.size(0)

        pred = torch.where(output > 0.5, 1, 0)
    
        evaluator.add_batch(target, pred)
        tbar.set_description('Training  ->>>- Epoch: [%3d]/[%3d]  Train loss: %.4f ' % (
            epoch+1, args.epochs, train_loss / total_num))
    result_dict = get_result_dict(evaluator, epoch, start_since, time.time(), train_loss, total_num)
    return result_dict

def validationing(val_loader, model, epoch, evaluator, args):
    val_loss = 0.0
    val_num = 0.0
    start_since = time.time()
    device = args.device
    model.eval()
    evaluator.reset()
    tbar = tqdm(val_loader, desc='valing>>>>>>>', leave=False)
    for i, batch in enumerate(tbar):
        x = batch[0]
        y = batch[1]

        rgb,  target= x.to(device),  y.to(device)
        rgb = F.interpolate(rgb, scale_factor=4, mode='bilinear')

        with torch.no_grad():
            output = model(rgb.float())
        loss = calc_loss(output, target.float())
        val_num +=  x.size(0)
        val_loss += loss.data.cpu().numpy() * x.size(0)

        pred = torch.where(output > 0.5, 1, 0) # 二值化
    
        evaluator.add_batch(target, pred)
        tbar.set_description('valing  ->>>- Epoch: [%3d]/[%3d]  val loss: %.4f ' % (
            epoch+1, args.epochs, val_loss / val_num))
    result_dict = get_result_dict(evaluator, epoch, start_since, time.time(), val_loss, val_num) 
    return result_dict

class jpeg_png_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,  train = True):
        if train:
            self.images_dir = data_dir + os.sep + 'images' + os.sep + 'train'
            self.labels_dir = data_dir + os.sep + 'labels' + os.sep + 'train'
            img_list    = sorted(os.listdir(self.images_dir))
            images = [i for i in img_list if i.endswith('.jpeg')]
            random.shuffle(images)
            self.images = images
        else:
            self.images_dir = data_dir + os.sep + 'images' + os.sep + 'val'
            self.labels_dir = data_dir + os.sep + 'labels' + os.sep + 'val'
            img_list    = sorted(os.listdir(self.images_dir))
            images = [i for i in img_list if i.endswith('.jpeg')]
            random.shuffle(images)
            self.images = images
        
        self.inp_size=1024

        self.img_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((self.inp_size, self.inp_size)),
        torchvision.transforms.ToTensor()
    ])

        self.mask_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((self.inp_size, self.inp_size)),
        torchvision.transforms.Grayscale(),  # 将图像转换为灰度模式
        torchvision.transforms.Lambda(lambda x: x.point(lambda p: p > 128 and 255)),  # 进行二值化操作
        torchvision.transforms.ToTensor(),
    ])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        PILimg = Image.open(self.images_dir + os.sep + self.images[i])

        label_arry = Image.open(self.labels_dir + os.sep + self.images[i][:-5] + '.png')
        # label_arry = np.array(label_arry).astype(np.uint8)
        # PILlabel = Image.fromarray(label_arry*255) 
        return self.img_transform(PILimg),  self.mask_transform(label_arry)#PILlabel)

import matplotlib.pyplot as plt
def display_images_with_predictions_and_labels(image1, prediction1, label1):
    fig, axs = plt.subplots(1, 3, figsize=(8, 6))

    axs[0].imshow(np.transpose(image1, (1,2,0)))
    axs[0].axis('off')
    axs[0].set_title('Image')

    axs[1].imshow(prediction1[0])
    axs[1].axis('off')
    axs[1].set_title('Prediction')

    axs[2].imshow(label1[0])
    axs[2].axis('off')
    axs[2].set_title('True Label')

    plt.tight_layout()
    plt.show()



def make_data_loaders(args):
    data_path = {"BJL":'/datasets/landslide_dataset/BJ_dataset',
        "L4S":'/datasets/L4S',
        "GVLM":'/datasets/GVLM'
    }[args.dataset]

    Training_Data = jpeg_png_Dataset(data_path, True)
    valing_Data   = jpeg_png_Dataset(data_path, False)

    print(f"Training_Data : {len(Training_Data)}    valing_Data: {len(valing_Data)}")
    train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=args.batch_size, shuffle=False,drop_last=True)#, collate_fn=collate_fn, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valing_Data, batch_size=args.batch_size, shuffle=False,drop_last=True)#,  collate_fn=collate_fn, shuffle=False) 
    return train_loader, valid_loader
