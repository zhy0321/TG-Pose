import matplotlib.pyplot as plt
import numpy as np


def vis_pointcloud(pc_org, pc_aug, cat_id, aug_name):
    fig = plt.figure()
    # fig, axes = plt.subplots(1, 2, projection='3d')
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    # points_category_GT_i = points_category_GT[i]
    # pc_org_i_trans = pc_org_i + 0.2

    difference = pc_org - pc_aug
    difference = np.sum(difference)
    # print(f'difference:', difference)

    ax1.scatter(pc_org[:, 0], pc_org[:, 1], pc_org[:, 2], c='b', marker='.')
    ax1.set_title('original pc')
    ax2.scatter(pc_aug[:, 0], pc_aug[:, 1], pc_aug[:, 2], c='r', marker='.')
    ax2.set_title('cat_id:{0},aug_name:{1}'.format(cat_id, aug_name))

    plt.show()
