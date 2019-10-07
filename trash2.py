import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


lab0 = np.load('ski30.npy').astype(np.uint8)
lab1 = np.load('ski31.npy').astype(np.uint8)
img_seg0 = lab0.copy()
img_seg0[img_seg0 != 0] = 1
img_seg1 = lab1.copy()
img_seg1[img_seg1 != 0] = 1

mu_theta = 3
std_theta = .2


def new_vec(v_f, v=None, cell_num=3, al=.7, be=.3):
    '''
    This function returns \alpha V + \beta V_f
    :param v_f: vector of flow
    :param v: movement vector of the cell if there is nothing it will
    make a random vector
    :param cell_num: number of cells
    :param al: alpha parameter of formula
    :param be: beta parameter of formula
    :return: \alpha V + \beta V_f
    '''
    if v is None:
        v = np.random.randint(-5, 6, (cell_num, 2))

    return al * v + be * v_f


def lab_move(labnd, v):
    '''
    This function moves cells of the labeled cells with respect to their
    related vector from v
    :param labnd: layered labeled binary cell. we have n = number of cell
    layer and in each layer we have binary image of related cell number
    :param v: vector of movement the cells. for each cell we have separate
    vector
    :return: labeled n dimentional image of cells.(labnd after
     vector v movemnet))
    '''
    ls = labnd.shape
    new_lab = np.zeros(ls)
    # temp_lab = np.zeros((ls[0], ls[1]))
    for num in range(ls[2]):
        if (np.where(labnd[:, :, num] == 1)[0]+int(v[num, 0]) >= ls[0]).any():
            new_lab[np.where(labnd[:, :, num] == 1)[0] - int(v[num, 0]),
                    np.where(labnd[:, :, num] == 1)[1] + int(v[num, 1]), num] = 1
        elif (np.where(labnd[:, :, num] == 1)[1]+int(v[num, 1]) >= ls[1]).any():
            new_lab[np.where(labnd[:, :, num] == 1)[0] + int(v[num, 0]),
                    np.where(labnd[:, :, num] == 1)[1] - int(v[num, 1]), num] = 1
        else:
            new_lab[np.where(labnd[:, :, num] == 1)[0] + int(v[num, 0]),
                    np.where(labnd[:, :, num] == 1)[1] + int(v[num, 1]), num] = 1

    return new_lab


def vect_correct(labnd, vec):
    '''
    correcting the direction of vectors that they caused overlap
    :param labnd: layered labeled binary cell. we have n = number of cell
    layer and in each layer we have binary image of related cell number
    :param vec: vector of movement the cells. for each cell we have separate
    vector
    :return: new vector after correcting the vector caused overlap.
    '''
    lab_sum = np.sum(labnd, axis=2)

    for n in range(len(vec)):
        if np.max(lab_sum[np.where(labnd[:, :, n] == 1)]) >= 3:
            vec[n] = -vec[n]
        elif np.max(lab_sum[np.where(labnd[:, :, n] == 1)]) == 2:
            cent0 = np.array([np.mean(np.where(layered_lab1[:, :, n])[0]),
                              np.mean(np.where(layered_lab1[:, :, n])[1])])

            eff_vecs = []
            for j in range(len(vec)):
                if j == n:
                    continue
                else:
                    if np.max(labnd[:, :, j][np.where(
                            labnd[:, :, n] == 1)]) > 0:
                        eff_vecs.append(vec[j])

                        cent1 = np.array([np.mean(np.where(
                            layered_lab1[:, :, j])[0]),
                                          np.mean(np.where(
                                              layered_lab1[:, :, j])[1])])

                        v_cent = cent0 - cent1
                        if np.linalg.norm(v_cent) > 10:
                            # print("It was v_cent for", n, " and ", j, "is ", v_cent)
                            v_cent = np.divide(v_cent, 10)
                        # print("v_cent for", n, " and ", j, "is ", v_cent)
                        eff_vecs.append(v_cent)

            vec[n] = np.mean(eff_vecs, axis=0)
            # print(vec[n])

    return vec


def cell_rotate(labnd, all_theta):
    '''
    This function rotate each cell with respect to related specified theta from all_theta
    :param labnd: layered labeled binary cell. we have n = number of cell
    :param all_theta: list of degree of rotation for each cell
    :return: layered labeled binary cell after rotation for related all_theta degrees
    '''
    for i in range(labnd.shape[2]):
        cell_slide = labnd[:, :, i]
        theta = all_theta[i]

        grid_x, grid_y = np.mgrid[0:cell_slide.shape[0], 0:cell_slide.shape[1]]
        cent = np.array([np.mean(np.where(cell_slide == 1)[0]), np.mean(np.where(cell_slide == 1)[1])])
        pts0 = np.where(cell_slide == 1)[0] - cent[0], np.where(cell_slide == 1)[1] - cent[1]
        xy = np.array([pts0[0] * np.cos(theta * np.pi / 180) -
                       pts0[1] * np.sin(theta * np.pi / 180),
                       pts0[0] * np.sin(theta * np.pi / 180) +
                       pts0[1] * np.cos(theta * np.pi / 180)]).T
        pts1 = np.array([xy[:, 0] + cent[0], xy[:, 1] + cent[1]]).T
        values = np.ones(np.shape(np.where(cell_slide == 1)[0]))
        img_interpolate = griddata(pts1, values, (grid_x, grid_y), method='linear')
        labnd[:, :, i] = np.uint8(np.matrix.round(img_interpolate))

    return labnd


l = list(lab0.shape)
l.append(np.max(lab0))

layered_lab0 = np.zeros((l[0], l[1], l[2]))

for n in range(np.max(lab0)):
    layered_lab0[np.where(lab0 == n + 1)[0], np.where(lab0 == n + 1)[1], n] = 1

plt.imshow(np.sum(layered_lab0, axis=2))
plt.show()
v = np.zeros((np.max(lab0), 2))
v[:, 0], v[:, 1] = 4, 3
nv = new_vec(v, cell_num=l[2])
layered_lab1 = lab_move(layered_lab0, nv)
plt.imshow(np.sum(layered_lab1, axis=2))
plt.show()
a_thet = mu_theta + std_theta * np.random.randn(l[2])
layered_lab1 = cell_rotate(layered_lab1, a_thet)
nv = vect_correct(layered_lab1, nv)
nv = new_vec(v, nv, cell_num=l[2])
layered_lab1 = lab_move(layered_lab1, nv)
plt.imshow(np.sum(layered_lab1, axis=2))
plt.show()
# for cou in range(11):
#     a_thet = mu_theta + std_theta * np.random.randn(l[2])
#     layered_lab1 = cell_rotate(layered_lab1, a_thet)
#     nv = vect_correct(layered_lab1, nv)
#     nv = new_vec(v, nv, cell_num=l[2])
#     layered_lab1 = lab_move(layered_lab1, nv)
#     if cou % 5 == 0:
#         plt.imshow(np.sum(layered_lab1, axis=2))
#         plt.show()

