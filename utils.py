import cv2
import numpy as np
import torch
import sqlite3
import os
import itertools

def image2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)

def get_init_cameraparams(width, height, modelId):
    f = max(width, height) * 1.2
    cx = width / 2.0
    cy = height / 2.0
    if modelId == 0:
        return np.array([f, cx, cy])
    elif modelId == 1:
        return np.array([f, f, cx, cy])
    elif modelId == 2 or modelId == 6:
        return np.array([f, cx, cy, 0.0])
    elif modelId == 3 or modelId == 7:
        return np.array([f, cx, cy, 0.0, 0.0])
    elif modelId == 4 or modelId == 8:
        return np.array([f, f, cx, cy, 0.0, 0.0, 0.0, 0.0])
    elif modelId == 9:
        return np.array([f, f, cx, cy, 0.0])
    return np.array([f, f, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def init_database(db):
    # path=os.path.join(paths,"men.db")
    # conn=sqlite3.connect(paths)
    # cursor=conn.cursor()
    cursor=db
    # sql="DELETE from two_view_geometries"
    # cursor.execute(sql)
    sql0="DELETE from descriptors"
    cursor.execute(sql0)
    sql1="DELETE from matches"
    cursor.execute(sql1)
    sq2="DELETE from keypoints"
    cursor.execute(sq2)
    # conn.commit()
    # cursor.close()
    # conn.close()
    # return 0

def image_to_pair_id(img_id):

    img_id_list=[]
    pair_id_dict={}
    for item in img_id:
        img_id_list.append(item)
    #图片两两不重复组合
    pair_list=list(itertools.combinations(img_id_list,2))
    #pair_id
    for pair in pair_list:
        image_id1=pair[0]
        image_id2=pair[1]
        if image_id1 > image_id2:
            pair_id_dict[2147483647 * image_id2 + image_id1]=[img_id[image_id2], img_id[image_id1]]
        else:
            pair_id_dict[2147483647 * image_id1 + image_id2]=[img_id[image_id1], img_id[image_id2]]
    
    return pair_id_dict