import numpy as np
import matplotlib
import cv2 as cv
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

no_of_Datasets = 2
Datasets = ['HAM10000', 'PH2Dataset']


def Image_Results():
    for i in range(no_of_Datasets):
        Orig = np.load('Images_' + str(i + 1) + '.npy', allow_pickle=True)
        Images = np.load('Preprocess_' + str(i + 1) + '.npy', allow_pickle=True)
        segment = np.load('segmentation_' + str(i + 1) + '.npy', allow_pickle=True)
        ground = np.load('GT_' + str(i + 1) + '.npy', allow_pickle=True)
        if i == 0:
            ind = [999, 1999, 2999, 3999, 4999]
        else:
            ind = [50, 65, 100, 150, 199]
        for j in range(len(ind)):
            original = Orig[ind[j]]
            image = Images[ind[j]].astype('uint8')
            seg = segment[ind[j]]
            GT = ground[ind[j]]

            fig, ax = plt.subplots(1, 3)
            plt.suptitle(Datasets[i], fontsize=20)
            plt.subplot(1, 4, 1)
            plt.title('Orig')
            plt.imshow(original)
            plt.subplot(1, 4, 2)
            plt.title('Prep')
            plt.imshow(image)
            plt.subplot(1, 4, 3)
            plt.title('Seg')
            plt.imshow(seg)
            plt.subplot(1, 4, 4)
            plt.title('GT')
            plt.imshow(GT)
            path1 = "./Results/Image_Results/Dataset_%s_Image_%s_image.png" % (i + 1, j + 1)
            plt.savefig(path1)
            plt.show()
            # cv.imwrite('./Results/Image_Results/Dataset-' + str(i+1) + 'orig-' + str(j + 1) + '.png', original)
            # cv.imwrite('./Results/Image_Results/Dataset-' + str(i+1) + 'preproc-' + str(j + 1) + '.png', image)
            # cv.imwrite('./Results/Image_Results/Dataset-' + str(i+1) + 'segment-' + str(j + 1) + '.png', seg)
            # cv.imwrite('./Results/Image_Results/Dataset-' + str(i + 1) + 'gt-' + str(j + 1) + '.png', GT)


if __name__ == '__main__':
    Image_Results()

