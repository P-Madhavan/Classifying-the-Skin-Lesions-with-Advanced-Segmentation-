import numpy as np
from prettytable import PrettyTable
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle
from sklearn import metrics
from sklearn.metrics import roc_curve


def Statistical(val):
    out = np.zeros((5))
    out[0] = max(val)
    out[1] = min(val)
    out[2] = np.mean(val)
    out[3] = np.median(val)
    out[4] = np.std(val)
    return out


def Plot_Confusion():
    no_of_Datasets = 2
    for n in range(no_of_Datasets):
        Eval = np.load('Eval_all.npy', allow_pickle=True)[n]
        value = Eval[4, 4, :5]
        val = np.asarray([0, 1, 1])
        data = {'y_Actual': [val.ravel()],
                'y_Predicted': [np.asarray(val).ravel()]
                }
        df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'][0], df['y_Predicted'][0], rownames=['Actual'],
                                       colnames=['Predicted'])
        value = value.astype('int')

        confusion_matrix.values[0, 0] = value[1]
        confusion_matrix.values[0, 1] = value[3]
        confusion_matrix.values[1, 0] = value[2]
        confusion_matrix.values[1, 1] = value[0]

        sn.heatmap(confusion_matrix, annot=True).set(title='Accuracy = ' + str(Eval[4, 4, 4] * 100)[:5] + '%')
        sn.plotting_context()
        path1 = './Results/Dataset_%s_Confusion.png' % (n + 1)
        plt.savefig(path1)
        plt.show()


def Plot_Results():
    Eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'FPR', 'FNR', 'FOR', 'NPV', 'FDR', 'F1-Score',
             'MCC']
    Graph_Term = [0, 5, 6, 7, 10]
    Classifier = ['TERMS', 'CNN', 'RESNET', 'ALEXNET', 'MS-MOBILENET', 'HCMMV3']

    no_of_Datasets = 2
    for n in range(no_of_Datasets):
        value = Eval[n, 4, :, 4:]  # Sigmoid Activation
        value[:, :-1] = value[:, :-1] * 100
        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[j, :])
        print('--------------------------------------------------', 'Dataset-', (n + 1), ' Comparison',
              '--------------------------------------------------')
        print(Table)

        Eval = np.load('Eval_all.npy', allow_pickle=True)
        for j in range(len(Graph_Term)):
            Graph = np.zeros((Eval.shape[1], Eval.shape[2]))
            for k in range(Eval.shape[1]):
                for l in range(Eval.shape[2]):
                    if Graph_Term[j] == 10:
                        Graph[k, l] = Eval[n, k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = Eval[n, k, l, Graph_Term[j] + 4] * 100

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(6)
            ax.bar(X + 0.00, Graph[:, 0], color='m', width=0.10, label="CNN")
            ax.bar(X + 0.10, Graph[:, 1], color='#cc9f3f', width=0.10, label="RESNET")
            ax.bar(X + 0.20, Graph[:, 2], color='#01f9c6', width=0.10, label="ALEXNET")
            ax.bar(X + 0.30, Graph[:, 3], color='#2000b1', width=0.10, label="MS-MOBILENET")
            ax.bar(X + 0.40, Graph[:, 4], color='#019529', width=0.10, label="HCMMV3")
            plt.xticks(X + 0.25, ('Linear', 'ReLU', 'Leaky ReLU', 'TanH', 'Sigmoid', 'Softmax'))
            plt.xlabel('Activation Functions')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc=1)
            path1 = "./Results/Dataset_%s__%s_bar.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

        Stacked_Terms = ['Precison_FDR', 'Recall_TNR', 'Specificity_FPR']
        for j in range(len(Stacked_Terms)):
            if j == 0:
                stack_counts = {'Precision': Eval[n, 4, :, 7] * 100, 'FDR': Eval[n, 4, :, 12] * 100, }
            elif j == 1:
                stack_counts = {'Recall': Eval[n, 4, :, 5] * 100,
                                'FNR': Eval[n, 4, :, 9] * 100, }  # Specificity is also called as TNR
            # elif j == 2:
            #     stack_counts = {'NPV': Eval[n, 4, :, 11] * 100, 'FOR': Eval[n, 4, :, 10] * 100, }
            else:
                stack_counts = {'Specificity': Eval[n, 4, :, 6] * 100, 'FPR': Eval[n, 4, :, 8] * 100, }
            width = 0.6  # the width of the bars: can also be len(x) sequence
            fig, ax = plt.subplots()
            bottom = np.zeros(5)
            l1 = np.zeros((2)).astype('str')
            for stack, count in stack_counts.items():
                if stack == 'Precision' or stack == 'Recall' or stack == 'NPV' or stack == 'Specificity':
                    p1 = plt.bar(Classifier[1:], count, width=width, color='deepskyblue', label=stack)
                    a = count
                    ax.bar_label(p1, label_type='center')
                else:
                    p2 = plt.bar(Classifier[1:], count, bottom=a, width=width, color='violet', label=stack)
                    ax.bar_label(p2, label_type='center')
            # ax.set_title('Epochs')
            path1 = "./Results/Dataset_%s_%s.png" % (n + 1, Stacked_Terms[j])
            plt.legend(loc=4)
            plt.savefig(path1)
            plt.show()


def Plot_ROC():
    lw = 2
    cls = ['CNN', 'RESNET', 'ALEXNET', 'MS-MOBILENET', 'HCMMV3']
    colors1 = cycle(["coral", "chocolate", "hotpink", "dodgerblue", "lime", ])
    colors2 = cycle(["hotpink", "plum", "chocolate", "navy", "red", "palegreen", "violet", "red"])
    for n in range(2):
        for i, color in zip(range(5), colors1):  # For all classifiers
            Predicted = np.load('roc_score.npy', allow_pickle=True)[n, 4, i].astype('float')
            Actual = np.load('roc_act.npy', allow_pickle=True)[n, 4, i].astype('int')
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[:, -1], Predicted[:, -1].ravel())

            auc = metrics.roc_auc_score(Actual[:, -1], Predicted[:,
                                                       -1].ravel())

            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label="{0}".format(cls[i]),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/roc_Dataset_%s.png" % (str(n + 1))
        plt.savefig(path1)
        plt.show()


def Plot_Segmentation():
    eval = np.load('Eval_Seg.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Dice', 'Jaccard']
    Methods = ['TERMS', 'Unet3+', 'ResUnet', 'ResUnet++', 'Mob-TransUnet+']
    for i in range(eval.shape[0]):
        value = eval[i, :, :, :]
        for j in range(len(Terms)):
            stats = np.zeros((len(Methods) - 1, 5))
            for k in range(len(Methods) - 1):
                stats[k, :] = Statistical(value[:, k, j])

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, stats[0, :], color='m', width=0.10, label="Unet3+")
            ax.bar(X + 0.10, stats[1, :], color='#cc9f3f', width=0.10, label="ResUnet")
            ax.bar(X + 0.20, stats[2, :], color='#01f9c6', width=0.10, label="ResUnet++")
            ax.bar(X + 0.30, stats[3, :], color='#2000b1', width=0.10, label=" Mobnetv3-TransUnet+")
            plt.xticks(X + 0.25, ('Best', 'Worst', 'Mean', 'Median', 'Std'))
            plt.xlabel('Statistical Analysis')
            plt.ylabel(Terms[j])
            plt.legend(loc=1)
            path1 = "./Results/Dataset_Seg_%s_%s_bar.png" % (i + 1, Terms[j])
            plt.savefig(path1)
            plt.show()


if __name__ == '__main__':
    Plot_Segmentation()
    Plot_ROC()
    Plot_Confusion()
    Plot_Results()
