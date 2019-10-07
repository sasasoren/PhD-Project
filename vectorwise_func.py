import cv2
import numpy as np
from conv_func import conv_fun, label_img,\
    col_cell, draw_flow
import scipy.io as sio
from statistics import mean
from skimage.measure import label, regionprops
from sklearn.metrics import accuracy_score
import random
import matplotlib.pyplot as plt
import sys
import compare_optical_flow as comp
from scipy.interpolate import griddata
from latex import insert_img, insert_img2, insert_img3,\
    img_table, insert_img3_onecap, img8_input
import conv_func as cf
from scipy.spatial import distance, Delaunay
from skimage.morphology import skeletonize
import time
from tqdm import tqdm
import itertools
import pandas as pd


save_add = "Images/mksample/"

par_var = [.1, .2, .5, 1, 2, 5, 10]
# par_var = 2.0**np.arange(-1, 11)
# par_var = [.5, 1.0, 8.0, 64, 512, 1024]

def cen_vec(img2d, save_name, save_ad=save_add, lab_img=None):
    # we have binary image 2 dimension as input and centroid and
    # vectors correspondent with all segments as output.
    if lab_img.all() == None:
        lab0 = label(img2d, connectivity=1)
    else:
        lab0 = lab_img
    centroid0 = np.zeros((np.max(lab0), 2))
    vec0 = np.zeros((np.max(lab0), 2))
    img_test0 = comp.for_show(img2d)
    for i in range(1, np.max(lab0) + 1):
        cell_img = np.zeros(np.shape(img2d))
        cell_img[lab0 == i] = 1
        skeleton0 = skeletonize(cell_img)
        y, x = np.where(skeleton0 == 1)
        centroid0[i - 1, 0] = np.int(np.mean(x))
        centroid0[i - 1, 1] = np.int(np.mean(y))
        A = list(zip(x, y))
        vec_arg = np.argmax(distance.cdist(centroid0[i - 1, :].
                                           reshape(1, 2),
                                           A, 'euclidean')[0])
        vec0[i - 1, 0] = A[vec_arg][0] - centroid0[i - 1, 0]
        vec0[i - 1, 1] = A[vec_arg][1] - centroid0[i - 1, 1]

        cv2.circle(img_test0, tuple(np.int32(centroid0[i - 1])),
                   2, (255, 0, 0))
        cv2.arrowedLine(img_test0,
                        tuple(np.int32(centroid0[i - 1]))
                        , A[vec_arg], (0, 0, 255), 1)

    # plt.imshow(img_test0)
    # plt.savefig(save_ad + save_name + ".png")
    # plt.show()

    return centroid0, vec0


def close_cen(centroid, point, win_size=15):
    cents = centroid.tolist()
    cents = list(filter(lambda w: np.all(
        w[0] <= point[0] + win_size), cents))
    cents = list(filter(lambda w: np.all(
        w[1] <= point[1] + win_size), cents))
    cents = list(filter(lambda w: np.all(
        w[0] >= point[0] - win_size), cents))
    cents = list(filter(lambda w: np.all(
        w[1] >= point[1] - win_size), cents))
    cent_list = []
    for cen in cents:
        cent_list.append(np.where((centroid == cen).
                                  all(axis=1))[0][0])

    return cents, cent_list


def choose_target(rel_dic, seg_num, taken_number):
    # This function gets relation dictionary and segment number
    # and current value of target and will return
    # new segment number as potential substitute for the last one
    poss_list = rel_dic[seg_num]
    poss_list = poss_list.tolist()
    poss_list.append(np.nan)
    for el in taken_number:
        poss_list.remove(el)

    if not poss_list:
        new_target = np.NaN
        print("{} is seg_num and {} is the taken_number".format(seg_num, taken_number))
    else:
        new_target = random.choice(poss_list)

    return new_target


def w_centroid(cent0, cent1):
    # This function will return norm squared of cent0-cent1
    w_i = np.linalg.norm((cent0 - cent1))
    w_sum = np.sum(w_i)
    return w_sum


def same_seg_delta(seg_num, t_seg, search_set):
    # t_seg should be just value of related dictionary
    # This function gets target function and count how many
    # same target segment exist in search set
    counter_ = 0
    for j in range(len(search_set)):
        if seg_num - 1 != j and t_seg == search_set[j]:
            counter_ += 1

    return counter_


def find_neighbors(pindex, triang):
    # This function will find neighbors of point pindex from
    # triang and return index of neighbors
    return triang.vertex_neighbor_vertices[1][
           triang.vertex_neighbor_vertices[0][pindex]:
           triang.vertex_neighbor_vertices[0][pindex+1]]


def neigh_dis(point, pind, pset, triangle, bound_dis=50):
    # This function gets pindex (index of desired point) and
    # coordinate of that and pset is the set of targeted set.
    # Output of the function is neighbors truncated from long
    # distance neighbors
    neig = find_neighbors(pind, triangle)
    neigh_set = neig.tolist()
    for neighbor in neig:
        if np.linalg.norm((point-pset[neighbor])) > bound_dis:
            neigh_set.remove(neighbor)

    A = [x+1 for x in neigh_set]
    A.append(np.nan)
    return A


def hm_target_neighbor(base_neigh, target_neigh, target_func):
    # This function counts the number of predicted cells which
    # they are not the new neighbors of the predicted target cell
    counter_ = 0
    for neigh in base_neigh:
        if neigh != neigh:
            counter_ = counter_
        elif target_func[neigh] not in target_neigh:
            counter_ += 1

    return counter_


def make_img(img_size, hm, save_ad, spd=6, rtn=80,
             d_rtn=10, lg_=(6, 11), d_lg=3, brd=20, split=False):
    # This function create two images as t=0 and 1 with size
    # img_size with hm steps and change in position with at
    # most speed of spd  and rotation of rtn, d_rt is the difference
    # between rtn0 and 1. brd is the border of image. lg: length growth.
    # b_size is the interval size

    img0 = np.zeros((img_size, img_size), np.uint8)
    img1 = np.zeros((img_size, img_size), np.uint8)

    img_lab0 = np.zeros((img_size, img_size), np.uint8)
    img_lab1 = np.zeros((img_size, img_size), np.uint8)

    num_ = np.int8(((img_size - 2*brd)/hm) + 1)
    xv, yv = np.meshgrid(np.arange(brd, img_size - brd, num_),
                         np.arange(brd, img_size - brd, num_))
    # crn is the number of cell in one layer almost equal to hm
    crn = len(np.arange(brd, img_size - brd, num_))
    rot0 = np.random.choice(np.arange(-rtn, rtn+1), (crn, crn))
    drt = np.random.choice(np.arange(-d_rtn, d_rtn+1), (crn, crn))
    rot1 = rot0 + drt
    minlg, maxlg = lg_
    cell_lg = np.random.choice(np.arange(minlg, maxlg), (crn, crn))
    dif_lg = np.random.choice(np.arange(d_lg), (crn, crn))
    lg_chng = cell_lg + dif_lg

    hor_chng = np.random.choice(np.arange(-spd, spd+1), crn)
    # hor_chng = np.zeros(crn, np.uint8)
    # hor_chng[0] = np.random.choice(np.arange(-spd, spd+1), 1, np.uint8)
    # for i in range(1, crn):
    #     temp = np.random.choice(np.arange(-spd, spd+1), 1, np.uint8)
    #     hor_chng[i] = hor_chng[i-1] + temp

    cent_chng = np.zeros((crn * crn, 2, 2), np.uint8)

    cnum = 1
    for ver in range(crn):
        ver_chng = np.random.choice(np.arange(-spd, spd+1))
        for hor in range(crn):
            cv2.ellipse(img0, (xv[ver, hor], yv[ver, hor]),
                        (cell_lg[ver, hor], 2), rot0[ver, hor], 0, 360, 255, -1)
            cv2.ellipse(img_lab0, (xv[ver, hor], yv[ver, hor]),
                        (cell_lg[ver, hor], 2),rot0[ver, hor], 0, 360, cnum, -1)

            cv2.ellipse(img1, (xv[ver, hor] + ver_chng, yv[ver, hor] +
                               hor_chng[hor]), (lg_chng[ver, hor], 2),
                        rot1[ver, hor], 0, 360, 255, -1)
            cv2.ellipse(img_lab1, (xv[ver, hor] + ver_chng, yv[ver, hor] +
                                   hor_chng[hor]), (lg_chng[ver, hor], 2),
                        rot1[ver, hor], 0, 360, cnum, -1)
            cent_chng[cnum - 1, 0, :] = xv[ver, hor], yv[ver, hor]
            cent_chng[cnum - 1, 1, :] = xv[ver, hor] + ver_chng,\
                                        yv[ver, hor] + hor_chng[hor]

            ver_chng += np.random.choice(np.arange(-spd, spd + 1))

            cnum += 1

        hor_chng += np.random.choice(np.arange(-spd, spd + 1), crn)

    if split == True:
        max_where = np.where(lg_chng == np.max(lg_chng))
        new_label = np.max(img_lab1)
        new_label_list = []
        for m in range(len(max_where[0])):
            max_lab = max_where[0][m] * crn + max_where[1][m] + 1
            new_label += 1
            # print(new_label)
            x, y = cent_chng[max_lab - 1, 1, :]
            x = x - 4
            y = y - 2
            img1[img_lab1 == max_lab] = 0
            img_lab1[img_lab1 == max_lab] = 0
            cv2.ellipse(img1, (x, y), (lg_[0], 2),
                        rot1[max_where[0][m], max_where[1][m]],
                        0, 360, 255, -1)
            cv2.ellipse(img_lab1, (x, y), (lg_[0], 2),
                        rot1[max_where[0][m], max_where[1][m]],
                        0, 360, int(new_label), -1)
            cv2.ellipse(img1, (x + 7, y + 9), (lg_[0], 2),
                        rot1[max_where[0][m], max_where[1][m]],
                        0, 360, 255, -1)
            cv2.ellipse(img_lab1, (x + 7, y + 9),
                        (lg_[0], 2), rot1[max_where[0][m], max_where[1][m]],
                        0, 360, int(new_label), -1)

            new_label_list.append([max_lab, new_label])
            # new_label_list returns label for both of the splited cells

        return img0, img1, img_lab0, img_lab1, cent_chng, new_label_list

    # cv2.imwrite(save_ad+"img0.png", img0)
    # cv2.imwrite(save_ad+"img1.png", img1)
    # cv2.imwrite(save_ad+"lab_img0.png", img_lab0)
    # cv2.imwrite(save_ad+"lab_img1.png", img_lab1)
    # np.save(save_ad+"cent_chng.npy", cent_chng)

    return img0, img1, img_lab0, img_lab1, cent_chng


def dict_to_list(diction, nrep=True):
    lst = []
    if nrep == True:
        for val in diction.values():
            for n in [val]:
                if n in lst:
                    continue
                else:
                    lst.append(n)
    else:
        for val in diction.values():
            for n in [val]:
                lst.append(n)

    return lst


def v_box(wch_, rel, vector0, vector1, targ, new=None, old=None):
    # v_box will return min of ||v_i - V_j|| for all cell j in moving window of
    # cell i from img0 and over all of j which they are not taken
    # by target function.
    # wch_: number of cell in img0 (aka i). rel: relation dictionary.
    # targ: target function.
    box = []
    lst = dict_to_list(targ)

    if old != None:
        lst.remove(old)
    elif new != None:
        lst.append(new)

    for nei in rel[wch_]:
        if nei == nei and nei not in lst:
            box.append(np.min((w_centroid(vector0[wch_ - 1], vector1[nei - 1]),
                               w_centroid(vector0[wch_ - 1], -vector1[nei - 1]))))

    if box == []:
        return 0
    else:
        return np.min(box)


def diff_cost(n_pred, o_pred, n_, cent0, cent1, v0, v1, neigh0, neigh1,
              avg_neigh, c_num, trgt_function, relat,
              alpha_=2, beta_=1000, gamma_=13, thrsh=1,
              alpha2_=2, alpha3_=.5, spl=.1):
    # In this function we are computing delta cost for our model.
    # n_pred: number of new prediction for the cell,
    # o_pred: number of old prediction for our model,
    # n_: number of cell in image0, relat: relation dictionary.
    # cent0, cent1: centroid for image 0 and 1,
    # v0, v1: correspondent vectors for each cell,
    # neigh0, neigh1: neighbors for all cells in image 0 ad 1,
    # avg_neigh: average number of neighbors,
    # c_num: total number of cell in image 0,
    # alpha_, beta_, gamma_, thrsh: parameters of th model
    # spl: is the rate of splitting cells
    if n_pred != n_pred:
        d_w = -w_centroid(cent0[n_ - 1], cent1[o_pred - 1])

        d_V = -np.min((w_centroid(v0[n_ - 1], v1[o_pred - 1]),
                      w_centroid(v0[n_ - 1], -v1[o_pred - 1])))

        d_same_seg = -same_seg_delta(n_, o_pred, list(trgt_function.values()))

        d_neighbor = len(neigh0[n_]) - 1 - hm_target_neighbor(neigh0[n_],
                                                              neigh1[o_pred],
                                                              trgt_function)

        d_star = np.max((0, v_box(n_, relat, v0, v1,
                                  trgt_function, old=o_pred) - thrsh))

        lst = dict_to_list(trgt_function, nrep=False)
        nan_num = np.count_nonzero(np.isnan(lst))
        d_perc = np.max((0, (nan_num + 1)/len(trgt_function) - spl)) -\
                 np.max((0, nan_num/len(trgt_function) - spl))

    elif o_pred != o_pred:
        d_w = w_centroid(cent0[n_ - 1], cent1[n_pred - 1])

        d_V = np.min((w_centroid(v0[n_ - 1], v1[n_pred - 1]),
                      w_centroid(v0[n_ - 1], -v1[n_pred - 1])))

        d_same_seg = same_seg_delta(n_, n_pred, list(trgt_function.values()))

        d_neighbor = hm_target_neighbor(neigh0[n_], neigh1[n_pred],
                                        trgt_function) - len(neigh0[n_]) - 1

        d_star = -np.max((0, v_box(n_, relat, v0, v1,
                                   trgt_function, new=n_pred) - thrsh))

        lst = dict_to_list(trgt_function, nrep=False)
        nan_num = np.count_nonzero(np.isnan(lst))
        d_perc = np.max((0, (nan_num - 1) / len(trgt_function) - spl)) - \
                 np.max((0, nan_num / len(trgt_function) - spl))

    else:
        # Here we are computing delta w in the formula
        d_w = w_centroid(cent0[n_ - 1], cent1[n_pred - 1]) - \
                 w_centroid(cent0[n_ - 1], cent1[o_pred - 1])
        # Here we are computing delta V in formula because we don't know
        # the direction of vectors we consider min of + and - as output
        d_V = np.min((w_centroid(v0[n_ - 1], v1[n_pred - 1]),
                          w_centroid(v0[n_ - 1], -v1[n_pred - 1]))) - \
                  np.min((w_centroid(v0[n_ - 1], v1[o_pred - 1]),
                          w_centroid(v0[n_ - 1], -v1[o_pred - 1])))

        d_same_seg = same_seg_delta(n_, n_pred, list(trgt_function.values())) - \
                     same_seg_delta(n_, o_pred, list(trgt_function.values()))

        d_neighbor = hm_target_neighbor(neigh0[n_], neigh1[n_pred],
                                        trgt_function) - hm_target_neighbor(
            neigh0[n_], neigh1[o_pred], trgt_function)

        d_star = 0

        d_perc = 0

    d_cost = ((d_w + alpha_ * d_V + beta_ * d_same_seg / (c_num - 2) +
               gamma_ * d_neighbor / avg_neigh -
               alpha2_ * d_star) / (c_num - 1)) + (alpha3_ * d_perc)

    return d_cost


def nan_or_not(dic):
    # This function separates cells with nan predict from not nan predicts
    nan_dic = {k: v for k, v in dic.items() if ~pd.Series(v).notna().all()}
    notna_dic = {k: v for k, v in dic.items() if pd.Series(v).notna().all()}
    return nan_dic, notna_dic


def v_cost(t_fun, v0, v1):
    # This function return cost of the differece of vectors part of the formula
    v_c = np.sum(np.min((np.linalg.norm(
        (v0[np.array(list(t_fun.keys())) - 1] +
         v1[np.array(list(t_fun.values())) - 1]), axis=1),
                         np.linalg.norm((v0[np.array(list(t_fun.keys())) - 1] -
                                         v1[np.array(list(t_fun.values())) - 1]),
                                        axis=1)), axis=0))
    return v_c


def same_seg_cost(t_fun):
    # finding cost of same segment part of the actual formula
    # t_fun is the target function
    s_cost = np.sum(list(map(lambda x: same_seg_delta(x, t_fun[x],
                                                      list(t_fun.values())),
                             list(t_fun.keys()))))
    return s_cost


def neighbor_cost(neigh0, neigh1, t_fun, notnan_t):
    # finding cost of neighbors part of the actual formula
    # neigh0 and 1 are neighbors for image 0 and 1
    # t_fun is target function
    # notnan_t is target function without nan parts
    neigh_cost = np.sum(list(map(lambda x: hm_target_neighbor(
        neigh0[x], neigh1[notnan_t[x]], t_fun),
                    list(notnan_t.keys()))))

    return neigh_cost


def total_cost(cen0, cen1, v0, v1, neigh0, neigh1, t_fun, rel, avg_neigh,
               c_num, alpha_=2, beta_=1000, gamma_=13, thrsh=1,
               alpha2_=2, alpha3_=.5, spl=.1):
    # finding total cost for each epoch and return separate cost and total cost as
    # output. Inputs are the same as d_cost.
    nan_target, notnan_target = nan_or_not(t_fun)
    w_cost = w_centroid(cen0[np.array(list(notnan_target.keys())) - 1],
                        cen1[np.array(list(notnan_target.values())) - 1])
    V_cost = v_cost(notnan_target, v0, v1)
    same_s_cost = same_seg_cost(t_fun)
    neigh_cost = neighbor_cost(neigh0, neigh1,
                               t_fun, notnan_target)
    star_cost = np.sum(list(map(lambda x: np.max(
        (0, v_box(x, rel, v0, v1, t_fun) - thrsh)),
                                     list(nan_target.keys()))))
    split_cost = np.max((0, (len(nan_target) - 1) /
                         len(t_fun) - spl))

    t_cost = ((w_cost + alpha_ * V_cost + beta_ * same_s_cost / (c_num - 2) +
               gamma_ * neigh_cost / avg_neigh -
               alpha2_ * star_cost) / (c_num - 1)) + (alpha3_ * split_cost)

    return w_cost, V_cost, same_s_cost, neigh_cost, star_cost, split_cost, t_cost


def find_target_function(img_seg0, img_seg1, lab0, lab1,
                         alpha_=8, beta_=1024, gamma_=64, thrsh=1,
                         alpha2_=1, alpha3_=1000, split_rate=.1,
                         iteration=100, T=100, dT=1/2):
    # This function gets two image img0 and img1 and related label images
    # beside that it needs parameters for formula and parameters
    # for boltzmann machine algorithm
    # it returns target function which is a function optimized to relate
    # segment labels from img0 to img1. target_function2
    # is the starting point
    centroid0, vec0 = cen_vec(img_seg0, save_name="vec0", lab_img=lab0)
    centroid1, vec1 = cen_vec(img_seg1, save_name="vec1", lab_img=lab1)

    relation = {}
    for i in range(1, np.max(lab0) + 1):
        _, cen_list = close_cen(centroid1,
                                centroid0[i - 1],
                                win_size=45)
        relation[i] = np.int32(cen_list) + 1

    target_function = {}
    for num in relation:
        target_function[num] = choose_target(relation, num, [])

    tri0 = Delaunay(centroid0)
    tri1 = Delaunay(centroid1)

    neighbors0 = {}
    neighbors1 = {}

    for i in range(1, np.max(lab0) + 1):
        neighbors0[i] = neigh_dis(centroid0[i - 1], i - 1, centroid0,
                                  tri0, bound_dis=60)

    for i in range(1, np.max(lab1) + 1):
        neighbors1[i] = neigh_dis(centroid1[i - 1], i - 1, centroid1,
                                  tri1, bound_dis=60)

    avg_neigh_num = mean([len(x) for x in neighbors0.values()])

    cell_num = len(target_function) + 1
    target_function2 = target_function.copy()

    costs = []

    for _ in tqdm(range(iteration)):
        for num_ in range(1, cell_num):
            old_pred = target_function[num_]
            new_pred = choose_target(relation, num_,
                                     [old_pred])

            delta_cost = diff_cost(new_pred, old_pred, num_,
                                   centroid0, centroid1, vec0, vec1,
                                   neighbors0, neighbors1,
                                   avg_neigh_num, cell_num,
                                   target_function, relation, alpha_=alpha_,
                                   beta_=beta_, gamma_=gamma_, thrsh=thrsh,
                                   alpha2_=alpha2_, alpha3_=alpha3_,
                                   spl=split_rate)

            if delta_cost < 0:
                target_function[num_] = new_pred
            else:
                u = np.random.uniform()
                if u < np.exp(-delta_cost / T):
                    target_function[num_] = new_pred

        costs.append(total_cost(centroid0, centroid1, vec0, vec1,
                                neighbors0, neighbors1, target_function,
                                relation, avg_neigh_num, cell_num,
                                alpha_=alpha_, beta_=beta_,
                                gamma_=gamma_, thrsh=thrsh,
                                alpha2_=alpha2_, alpha3_=alpha3_,
                                spl=split_rate))

        T = T * dT

    return target_function, target_function2

#
# lab0 = np.load('ski30.npy').astype(np.uint8)
# lab1 = np.load('ski31.npy').astype(np.uint8)
# img_seg0 = lab0.copy()
# img_seg0[img_seg0 != 0] = 1
# img_seg1 = lab1.copy()
# img_seg1[img_seg1 != 0] = 1
#
#
# alpha_, beta_, gamma_, thrsh, alpha2_, alpha3_ = 8, 1024, 64, 1, 1, 1000
# split_rate = .1
# T = 100
# dT = 1/2
# iteration = 100
#
# tar1, tar2 = find_target_function(img_seg0, img_seg1, lab0, lab1)

# img0, img1, img_lab0, img_lab1, cent_chng, new_label_list = make_img(
#     200, 7, save_add, spd=4, rtn=40,
#     d_rtn=10, lg_=(6, 11), d_lg=3, brd=20, split=True)
#
# plt.imshow(img0, cmap=plt.cm.gray, interpolation='nearest')
# plt.axis('off')
# plt.show()
# plt.imshow(img1, cmap=plt.cm.gray, interpolation='nearest')
# plt.axis('off')
# plt.show()




# lab_0 = np.load('ski30.npy').astype(np.uint8)
# lab_1 = np.load('ski31.npy').astype(np.uint8)
# img_0 = lab_0.copy()
# img_0[img_0 != 0] = 1
# img_1 = lab_1.copy()
# img_1[img_1 != 0] = 1

# t_fun = vectorwise_bm(img_0, img_1, lab_0, lab_1,
#                       alpha_, beta_, gamma_, thrsh, alpha2_, alpha3_, T=100)

# cnt = 0
# for num_ in range(1, cell_num):
#     if target_function[num_] == num_:
#         cnt += 1

    # print("accuracy is equal : ", cnt / (cell_num-1))

    # total_table[total_counter, 0] = alpha_
    # total_table[total_counter, 1] = beta_
    # total_table[total_counter, 2] = gamma_
    # total_table[total_counter, 3] = thrsh
    # total_table[total_counter, 4] = alpha2_
    # total_table[total_counter, 5] = alpha3_
    # total_table[total_counter, 6] = cnt / (cell_num-1)
    #
    # total_counter += 1
    # print("total counter:", total_counter)

# end = time.time()
#
# # print("accuracy is equal : ", cnt/cell_num)
# print("computational time is: ", end-start)



#
# img0 = np.zeros((200, 200), np.uint8)
# img1 = np.zeros((200, 200), np.uint8)
# lab_img0 = np.zeros((200, 200), np.uint8)
# lab_img1 = np.zeros((200, 200), np.uint8)
# xv, yv = np.meshgrid(np.arange(20, 180, 20),
#                      np.arange(20, 180, 14))
#
# adaad = [9, 12, 15, 16, 21, 22, 25, 27, 30, 41, 43, 48, 66, 72, 90]
# dovom = 96
# for adad in adaad:
#     dovom += 1
#     x, y = cent_chng[adad - 1, 1, :]
#     x = x - 4
#     y = y - 2
#     img1[lab_img1 == adad] = 0
#     lab_img1[lab_img1 == adad] = 0
#     cv2.ellipse(img1, (x, y), (6, 2), 35, 0, 360, 255, -1)
#     cv2.ellipse(lab_img1, (x, y), (6, 2), 35, 0, 360, adad, -1)
#     cv2.ellipse(img1, (x + 7, y + 9), (6, 2), 35, 0, 360, 255, -1)
#     cv2.ellipse(lab_img1, (x + 7, y + 9), (6, 2), 35, 0, 360, dovom, -1)
#
# save_ad = "Images/mksample/"
# cv2.imwrite(save_ad + "img000.png", img0)
# cv2.imwrite(save_ad + "img111.png", img1)
# cv2.imwrite(save_ad + "lab_img000.png", lab_img0)
# cv2.imwrite(save_ad + "lab_img111.png", lab_img1)



#
# xdix1 = img_seg0.copy()
# xdix2 = img_seg1.copy()
#
# xdix1 = cf.label_img(comp.for_show(xdix1), xdix1)
# xdix2 = cf.label_img(comp.for_show(xdix2), xdix2)
#
# plt.imshow(xdix1)
# plt.savefig("Images/vec_optim/xdix0.png")
# plt.show()
# plt.imshow(xdix2)
# plt.savefig("Images/vec_optim/xdix1.png")
# plt.show()


















# lab0 = label(img_seg0, connectivity=1)
# lab1 = label(img_seg1, connectivity=1)

# img2_seg0 = img_seg0.copy()
# img2_seg1 = img_seg1.copy()
#
# img_test0 = comp.for_show(img_seg0)
# img_test1 = comp.for_show(img_seg1)
# skeleton0 = skeletonize(img2_seg0/255)
# skeleton1 = skeletonize(img2_seg1/255)





# centroid0 = np.zeros((np.max(lab0), 2))
# vec0 = np.zeros((np.max(lab0), 2))
# for i in range(1, np.max(lab0) + 1):
#     cell_img = np.zeros(np.shape(img_seg0))
#     cell_img[lab0 == i] = 1
#     skeleton0 = skeletonize(cell_img)
#     y, x = np.where(skeleton0 == 1)
#     centroid0[i - 1, 0] = np.int(np.mean(x))
#     centroid0[i - 1, 1] = np.int(np.mean(y))
#     A = list(zip(x, y))
#     vec_arg = np.argmax(distance.cdist(centroid0[i - 1, :].
#                                        reshape(1, 2),
#                                        A, 'euclidean')[0])
#     vec0[i - 1, 0] = centroid0[i - 1, 0] - A[vec_arg][0]
#     vec0[i - 1, 1] = centroid0[i - 1, 1] - A[vec_arg][1]
#     cv2.circle(img_test0, tuple(np.int32(centroid0[i - 1])),
#                2, (255, 0, 0))
#     cv2.arrowedLine(img_test0, tuple(np.int32(centroid0[i - 1]))
#                     , A[vec_arg], (0, 0, 255), 1)
#
# centroid1 = np.zeros((np.max(lab1), 2))
# vec1 = np.zeros((np.max(lab1), 2))
# for i in range(1, np.max(lab1)+1):
#     cell_img = np.zeros(np.shape(img_seg1))
#     cell_img[lab1 == i] = 1
#     skeleton1 = skeletonize(cell_img)
#     y, x = np.where(skeleton1 == 1)
#     centroid1[i - 1, 0] = np.int(np.mean(x))
#     centroid1[i - 1, 1] = np.int(np.mean(y))
#     cv2.circle(img_test1, tuple(np.int32(centroid1[i - 1])),
#                2, (255, 0, 0))
#     A = list(zip(x, y))
#     vec_arg = np.argmax(distance.cdist(centroid1[i - 1, :]
#                                        .reshape(1, 2),
#                                        A, 'euclidean')[0])
#     vec1[i - 1, 0] = A[vec_arg][0] - centroid1[i - 1, 0]
#     vec1[i - 1, 1] = A[vec_arg][1] - centroid1[i - 1, 1]
#     cv2.arrowedLine(img_test1, tuple(np.int32(centroid1[i - 1]))
#                     , A[vec_arg], (0, 0, 255), 1)





























# lab0 = label(img_seg0, connectivity=1)
# lab1 = label(img_seg1, connectivity=1)

# img2_seg0 = img_seg0.copy()
# img2_seg1 = img_seg1.copy()
#
# img_test0 = comp.for_show(img_seg0)
# img_test1 = comp.for_show(img_seg1)
# skeleton0 = skeletonize(img2_seg0/255)
# skeleton1 = skeletonize(img2_seg1/255)





# centroid0 = np.zeros((np.max(lab0), 2))
# vec0 = np.zeros((np.max(lab0), 2))
# for i in range(1, np.max(lab0) + 1):
#     cell_img = np.zeros(np.shape(img_seg0))
#     cell_img[lab0 == i] = 1
#     skeleton0 = skeletonize(cell_img)
#     y, x = np.where(skeleton0 == 1)
#     centroid0[i - 1, 0] = np.int(np.mean(x))
#     centroid0[i - 1, 1] = np.int(np.mean(y))
#     A = list(zip(x, y))
#     vec_arg = np.argmax(distance.cdist(centroid0[i - 1, :].
#                                        reshape(1, 2),
#                                        A, 'euclidean')[0])
#     vec0[i - 1, 0] = centroid0[i - 1, 0] - A[vec_arg][0]
#     vec0[i - 1, 1] = centroid0[i - 1, 1] - A[vec_arg][1]
#     cv2.circle(img_test0, tuple(np.int32(centroid0[i - 1])),
#                2, (255, 0, 0))
#     cv2.arrowedLine(img_test0, tuple(np.int32(centroid0[i - 1]))
#                     , A[vec_arg], (0, 0, 255), 1)
#
# centroid1 = np.zeros((np.max(lab1), 2))
# vec1 = np.zeros((np.max(lab1), 2))
# for i in range(1, np.max(lab1)+1):
#     cell_img = np.zeros(np.shape(img_seg1))
#     cell_img[lab1 == i] = 1
#     skeleton1 = skeletonize(cell_img)
#     y, x = np.where(skeleton1 == 1)
#     centroid1[i - 1, 0] = np.int(np.mean(x))
#     centroid1[i - 1, 1] = np.int(np.mean(y))
#     cv2.circle(img_test1, tuple(np.int32(centroid1[i - 1])),
#                2, (255, 0, 0))
#     A = list(zip(x, y))
#     vec_arg = np.argmax(distance.cdist(centroid1[i - 1, :]
#                                        .reshape(1, 2),
#                                        A, 'euclidean')[0])
#     vec1[i - 1, 0] = A[vec_arg][0] - centroid1[i - 1, 0]
#     vec1[i - 1, 1] = A[vec_arg][1] - centroid1[i - 1, 1]
#     cv2.arrowedLine(img_test1, tuple(np.int32(centroid1[i - 1]))
#                     , A[vec_arg], (0, 0, 255), 1)




