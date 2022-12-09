from argparse import ArgumentParser
import utils
import numpy as np
import matplotlib.pyplot as plt
from experiment import Experiment

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--models", type=str, default="baseline", dest="models")
    parser.add_argument("--epoch", type=int, default=50, dest="epoch")
    parser.add_argument("--test_step", type=int, default=5, dest="test_step")
    args = parser.parse_args()

    models = args.models.split(',')
    epoch = args.epoch
    test_step = args.test_step
    x = np.arange(start=test_step, stop=epoch + 1, step=test_step)
    utils.make_dir('./plot')

    CMC = []
    mAP = []

    for model in models:
        e = Experiment(selected_model=model, epoch=epoch, test_step=test_step)
        e.train()

        CMC.append(e.track_CMC)
        mAP.append(e.track_mAP)

    print("\n\nfinal results\n")
    for i in range(len(models)):
        model = models[i]
        print("{}, rank-1:{:.4f}, rank-5:{:.4f}, rank-10:{:.4f}, mAP:{:.4f}".format(model, CMC[i][-1][0], CMC[i][-1][4],
                                                                                    CMC[i][-1][9], mAP[i][-1]))

    rank_x = np.arange(10) + 1
    for i in range(len(models)):
        plt.plot(rank_x, CMC[i][-1][:10], label=models[i])
    plt.xlabel('rank-k')
    plt.ylabel('accuracy')
    plt.title('CMC-k')
    plt.legend()
    plt.savefig("./plot/CMC.jpg")

    plt.clf()
    for i in range(len(models)):
        plt.plot(x, mAP[i], label=models[i])
    plt.xlabel('epoch')
    plt.ylabel('mAP')
    plt.title('mAP Over Epochs')
    plt.legend()
    plt.savefig("./plot/mAP.jpg")
