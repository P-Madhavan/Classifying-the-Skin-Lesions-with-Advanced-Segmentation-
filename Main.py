import os
import cv2 as cv
import numpy as np
import pandas as pd
from Global_vars import Global_vars
from Image_Results import Image_Results
from Model_MobTransUnetPlus import Model_MobTransUnetPlus
from Model_ResUnetPlusPlus import Model_ResUnetPlusPlus
from Model_ResUnet import Model_ResUnet
from Model_Unet3Plus import Model_Unet3Plus
from Model_CNN import Model_CNN
from Model_Resnet import Model_Resnet
from Model_Alexnet import Model_Alexnet
from Model_MS_Mobnet import Model_MS_Mobnet
from Model_HCMMV3 import Model_HCMMV3
from Plot_Results import Plot_Segmentation, Plot_ROC, Plot_Confusion, Plot_Results

no_of_Datasets = 2


def ReadText(filename):
    f = open(filename, "r")
    lines = f.readlines()
    Tar = []
    fileNames = []
    for lineIndex in range(len(lines)):
        if lineIndex and '||' in lines[lineIndex]:
            line = [i.strip() for i in lines[lineIndex].strip().strip('||').replace('||', '|').split('|')]
            fileNames.append(line[0])
            Tar.append(int(line[2]))
    Tar = np.asarray(Tar)
    uniq = np.unique(Tar)
    Target = np.zeros((len(Tar), len(uniq))).astype('int')
    for i in range(len(uniq)):
        index = np.where(Tar == uniq[i])
        Target[index, i] = 1
    return fileNames, Target


def Read_Image(Filename, Gr=0):
    image = cv.imread(Filename)
    image = np.uint8(image)
    if Gr == 1:
        if len(image.shape) == 3:
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, (224, 224))
    return image


def Read_Images(Directory):
    Images = []
    out_folder = os.listdir(Directory)
    for i in range(len(out_folder)):
        print(i)
        filename = Directory + out_folder[i]
        image = Read_Image(filename)
        Images.append(image)
    return Images


def Read_Datset_PH2(Directory, fileNames):
    Images = []
    GT = []
    folders = os.listdir(Directory)
    for i in range(len(folders)):
        if folders[i] in fileNames:
            image = Read_Image(Directory + folders[i] + '/' + folders[i] + '_Dermoscopic_Image/' + folders[i] + '.bmp')
            gt = Read_Image(Directory + folders[i] + '/' + folders[i] + '_lesion/' + folders[i] + '_lesion.bmp', Gr=1)
            Images.append(image)
            GT.append(gt)
    return Images, GT


def Read_CSV(Path):
    df = pd.read_csv(Path)
    values = df.to_numpy()
    value = values[:, 6]
    uniq = np.unique(value)
    Target = np.zeros((len(value), len(uniq))).astype('int')
    for i in range(len(uniq)):
        index = np.where(value == uniq[i])
        Target[index, i] = 1
    return Target


# Read Datasets
an = 0
if an == 1:
    Images1 = Read_Images('./Datasets/HAM10000/Images/')
    np.save('Images_1.npy', Images1)

    Target1 = Read_CSV('./Datasets/HAM10000/HAM10000_metadata.csv')
    np.save('Target_1.npy', Target1)

    fileNames, Target2 = ReadText('./Datasets/PH2Dataset/PH2_dataset.txt')
    Images2, GT = Read_Datset_PH2('./Datasets/PH2Dataset/PH2 Dataset images/', fileNames)
    np.save('Images_2.npy', Images2)
    np.save('GT_2.npy', GT)
    np.save('Target_2.npy', Target2)

# GroundTruth for Dataset1
an = 0
if an == 1:
    im = []
    Img = np.load('Images_1.npy', allow_pickle=True)
    for i in range(len(Img)):
        image = Img[i]
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        minimum = int(np.min(image))
        maximum = int(np.max(image))
        Sum = ((minimum + maximum) / 2)
        ret, thresh = cv.threshold(image, Sum, 255, cv.THRESH_BINARY_INV)
        im.append(thresh)
    np.save('GT_1.npy', im)

# pre-processing #
an = 0
if an == 1:
    for i in range(no_of_Datasets):
        Img = np.load('Images_' + str(i + 1) + '.npy', allow_pickle=True)
        Preprocess = []
        for j in range(len(Img)):
            Orig = Img[j]
            # applying the median filter
            img = cv.medianBlur(Orig, 3)

            # The declaration of CLAHE
            # clipLimit -> Threshold for contrast limiting
            clahe = cv.createCLAHE(clipLimit=2)
            imge_cl = np.zeros((img.shape)).astype('uint8')
            imge_cl[:, :, 0] = clahe.apply(img[:, :, 0])
            imge_cl[:, :, 1] = clahe.apply(img[:, :, 1])
            imge_cl[:, :, 2] = clahe.apply(img[:, :, 2])

            Preprocess.append(imge_cl)
        np.save('Preprocess_' + str(i + 1) + '.npy', Preprocess)

## Lesion segmentation using - Mobilenetv3 based TransUnet+
an = 0
if an == 1:
    Eval_Seg = []
    for i in range(no_of_Datasets):
        Img = np.load('Preprocess_' + str(i + 1) + '.npy', allow_pickle=True)
        Gt = np.load('GT_' + str(i + 1) + '.npy', allow_pickle=True)
        Ev = []

        per = round(Img.shape[0] * 0.75)  # % of learning
        train_images = Img[:per]
        train_masks = Gt[:per]
        test_images = Img[per:]
        test_masks = Gt[per:]

        Eval = Model_Unet3Plus(train_images, train_masks, test_images, test_masks, Img)  # Unet3+
        Ev.append(Eval)
        Eval = Model_ResUnet(train_images, train_masks, test_images, test_masks, Img)  # ResUnet
        Ev.append(Eval)
        Eval = Model_ResUnetPlusPlus(train_images, train_masks, test_images, test_masks, Img)  # ResUnet++
        Ev.append(Eval)
        Im, Eval = Model_MobTransUnetPlus(train_images, train_masks, test_images, test_masks,
                                          Img)  # Mobilenetv3 based TransUnet+
        np.save('segmentation_' + str(i + 1) + '.npy', Im)
        Ev.append(Eval)

        Eval_Seg.append(Ev)
    np.save('Eval_Seg.npy', Eval_Seg)

# classification
an = 0
if an == 1:
    Eval_all = []
    Actn = ['Linear', 'ReLU', 'Leaky ReLU', 'TanH', 'Sigmoid', 'Softmax']
    for n in range(no_of_Datasets):
        Features = np.load('segmentation_' + str(n + 1) + '.npy', allow_pickle=True)[:100]
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)[:100]
        learnperc = round(Features.shape[0] * 0.75)

        Train_Data = Features[:learnperc, :, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Features[learnperc:, :, :]
        Test_Target = Target[learnperc:, :]

        Ev = []
        for i in range(len(Actn)):
            Global_vars.activation = Actn[i]
            Eval = np.zeros((5, 14))
            Eval[0, :] = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[1, :] = Model_Resnet(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[2, :] = Model_Alexnet(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[3, :] = Model_MS_Mobnet(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[4, :] = Model_HCMMV3(Train_Data, Train_Target, Test_Data, Test_Target)
            Ev.append(Eval)
        Eval_all.append(Ev)
    np.save('Eval_all.npy', Eval_all)

Image_Results()
Plot_Segmentation()
Plot_ROC()
Plot_Confusion()
Plot_Results()
