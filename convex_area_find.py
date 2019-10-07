import numpy as np
import pickle
import vectorwise_func2 as vf2
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from statistics import mean
from tqdm import tqdm


relation_window_size = 45
change_matrix_list = []
img_num = 1

for img_num in range(2):
    lab0 = np.load('Images/moving_simul/w/{}/lab0.npy'.format(img_num)).astype(np.uint8)
    lab1 = np.load('Images/moving_simul/w/{}/lab1.npy'.format(img_num)).astype(np.uint8)
    # with open('Images/moving_simul/{}/rel_dic.pkl'.format(img_num), 'rb') as f:
    #     s_dic = pickle.load(f)

    img_seg0 = lab0.copy()
    img_seg0[img_seg0 != 0] = 1
    img_seg1 = lab1.copy()
    img_seg1[img_seg1 != 0] = 1
    # al = [8, 500, 2, 1, 1000, 8, .01, 500, 500]

    com_d = vf2.small_dic_to_complete({}, np.max(lab0))

    centroid0, vec0 = vf2.cen_vec(img_seg0, save_name="vec0", lab_img=lab0)
    centroid1, vec1 = vf2.cen_vec(img_seg1, save_name="vec1", lab_img=lab1)

    relation = {}
    for i in range(1, np.max(lab0) + 1):
        _, cen_list = vf2.close_cen(centroid1,
                                    centroid0[i - 1],
                                    win_size=relation_window_size)
        relation[i] = np.int32(cen_list) + 1

    tri0 = Delaunay(centroid0)
    tri1 = Delaunay(centroid1)

    neighbors0 = {}
    neighbors1 = {}

    neighbors0_nonan = {}
    neighbors1_nonan = {}

    for i in range(1, np.max(lab0) + 1):
        neighbors0[i], neighbors0_nonan[i] = vf2.neigh_dis(centroid0[i - 1], i - 1, centroid0,
                                                           tri0, bound_dis=80)

    for i in range(1, np.max(lab1) + 1):
        neighbors1[i], neighbors1_nonan[i] = vf2.neigh_dis(centroid1[i - 1], i - 1, centroid1,
                                                           tri1, bound_dis=80)

    avg_neigh_num = mean([len(x) for x in neighbors0.values()])

    # first_cost = vf2.total_cost(centroid0, centroid1, vec0, vec1, neighbors0,
    # neighbors1, com_d, relation,
    #                             avg_neigh_num, al, spl=.1, tsh=1)

    first_cost = vf2.new_parameter_cost_no_split(centroid0, centroid1, vec0, vec1,
                                                       neighbors0_nonan, neighbors1_nonan,
                                                       com_d, avg_neigh_num)
    print('first change {}:'.format(img_num), first_cost)

    # change_matrix = np.zeros((np.max(lab0), len(first_cost)))
    # # change_matrix = np.zeros((100, len(first_cost)))
    # new_dic = com_d.copy()
    #
    # for cell_num in range(1, np.max(lab0) + 1):
    # # for cell_num in range(1, 101):
    #     temp_dic = com_d.copy()
    #     temp_dic[cell_num] = vf2.choose_target_no_split(relation, cell_num, temp_dic[cell_num])
    #     # temp_dic = vf2.make_random_target_no_split(relation)
    #     if temp_dic[cell_num] == temp_dic[cell_num]:
    #         new_dic = temp_dic.copy()
    #         second_cost = vf2.new_parameter_cost_no_split(centroid0, centroid1, vec0, vec1,
    #                                                             neighbors0_nonan, neighbors1_nonan,
    #                                                             temp_dic, avg_neigh_num)
    #
    #         change_matrix[cell_num - 1, :] = np.array(second_cost) - np.array(first_cost)
    #     else:
    #         change_matrix[cell_num - 1, :] = 0, 0, 0

    # change_matrix = np.zeros((np.max(lab0), len(first_cost)))
    change_matrix = np.zeros((1000, len(first_cost)))
    # new_dic = com_d.copy()
    #
    # for cell_num in range(1, np.max(lab0) + 1):
    for cell_num in range(1, 1001):
        # temp_dic = com_d.copy()
        # temp_dic[cell_num] = vf2.choose_target_no_split(relation, cell_num, temp_dic[cell_num])
        temp_dic = vf2.make_random_target_no_split(relation)
        # if temp_dic[cell_num] == temp_dic[cell_num]:
        #     new_dic = temp_dic.copy()
        second_cost = vf2.new_parameter_cost_no_split(centroid0, centroid1, vec0, vec1,
                                                            neighbors0_nonan, neighbors1_nonan,
                                                            temp_dic, avg_neigh_num)

        change_matrix[cell_num - 1, :] = np.array(second_cost) - np.array(first_cost)
        # else:
        #     change_matrix[cell_num - 1, :] = 0, 0, 0
    normalized_change = (change_matrix - np.mean(change_matrix, axis=0))
                        # / np.std(change_matrix, axis=0)
    change_matrix_list.append(normalized_change)
    # change_matrix_list.append(change_matrix)

all_changes = np.concatenate(change_matrix_list, axis=0)

















