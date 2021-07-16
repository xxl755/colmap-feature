import sqlite3
from colmap_db import import_db
from database import COLMAPDatabase
import os
import cv2
import numpy as np
from utils import get_init_cameraparams,init_database,image_to_pair_id
import time


camModelDict = {'SIMPLE_PINHOLE': 0,
                'PINHOLE': 1,
                'SIMPLE_RADIAL': 2,
                'RADIAL': 3,
                'OPENCV': 4,
                'FULL_OPENCV': 5,
                'SIMPLE_RADIAL_FISHEYE': 6,
                'RADIAL_FISHEYE': 7,
                'OPENCV_FISHEYE': 8,
                'FOV': 9,
                'THIN_PRISM_FISHEYE': 10}

def operate(cmd):
    print(cmd)
    start = time.perf_counter()
    os.system(cmd)#删除.db文件
    end = time.perf_counter()
    duration = end - start
    print("[%s] cost %f s" % (cmd, duration))

def init_db_camera(db,img_path,cam_model,single_flag):
    modelid=camModelDict[cam_model]

    widths=[]
    heights=[]
    init_params=[]
    for  img_name in sorted(os.listdir(img_path)):
        img_full_path=os.path.join(img_path,img_name)
        img=cv2.imread(img_full_path)
        height,width=img.shape[:2]
        widths.append(width)
        heights.append(height)
        init_param=get_init_cameraparams(width,height,modelid)
        init_params.append(init_param)

    if single_flag:
        db.add_camera(modelid,widths[0],heights[0],init_params[0],camera_id=0)
    else:
        for camera_idx in len(sorted(os.listdir(img_path))):
            db.add_camera(modelid,widths[camera_idx],heights[camera_idx],camera_id=camera_idx)
    # db.commit()

    img_dict={}
    id_dict={}
    for img_idx,img_name in enumerate(sorted(os.listdir(img_path))):
        if single_flag:
            db.add_image(img_name, camera_id=0,image_id=img_idx)
            
        else:
            db.add_image(img_name,camera_id=img_idx,image_id=img_idx)

        img_dict[img_idx]=img_name
        id_dict[img_name]=img_idx
    # db.commit()

    return img_dict,id_dict

def add_features(paths,img_dict,db):

    init_database(db)
    match_info_dict={}
    for img_id,image in img_dict.items():

        kps_dict={}
        des_dict={}
        
        image_path=os.path.join(paths+"/imagetest",image)
        img=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
   
        sift=cv2.SIFT_create(13000)

        kp_s,des_s=sift.detectAndCompute(img,None)

        des_s=np.array(des_s)
        kp_list=[]
        for kp in kp_s:
            kp_list.append([kp.pt[0],kp.pt[1]])
            #保留xy坐标
        kp_list=np.array(kp_list)#.reshape(-1,4)
        kp_list=kp_list[:,:2]

        des_dict["des"]=des_s
        kps_dict["kps"]=kp_list
        match_info_dict[image]=des_dict,kps_dict
        kp_data=np.concatenate([kp_list,np.ones([kp_list.shape[0],1]),np.zeros([kp_list.shape[0],1])],axis=1).astype(np.float32)
        db.add_keypoints(img_id,kp_data)
        db.add_descriptors(img_id,des_s)
    
    # db.commit()
    return match_info_dict

def get_matches(db,match_info_dict,pair_id,match_list_path,id_dict):


    match_dict={}

    match_list = open(match_list_path, 'w')

    for pairs in pair_id.values():
        match_list.write("%s %s\n" % (pairs[0], pairs[1]))

    match_list.close()

    FLANN_INDEX_KDTREE=0
    indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    searchParams= dict(checks=50)

    for id, img_pair in pair_id.items():
        des1=match_info_dict[img_pair[0]][0]["des"]
        des2=match_info_dict[img_pair[1]][0]["des"]
        flann=cv2.FlannBasedMatcher(indexParams,searchParams)
        matches=flann.knnMatch(des1,des2,2)
        match_info=[]

        idx=0
        while idx<len(matches):
            #去除错误匹配
            m,n=matches[idx]
            if m.distance>0.8*n.distance:#两个相近的匹配要距离差距足够大才保留
                matches.pop(idx)
               
            else:
                # print(matches[idx][0].distance
                match_info.append((matches[idx][0].distance,matches[idx][0].queryIdx,matches[idx][0].trainIdx))#保留匹配最好的数据
                idx+=1
            
        match_info=np.array(match_info)[:,1:].astype(np.int16)#按照距离排序，并去除距离信息\
        db.add_matches(id_dict[img_pair[0]],id_dict[img_pair[1]],match_info)
        # db.commit()
        match_dict[id]=match_info


    return match_dict

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mapper(projpath, images_path):
    database_path = os.path.join(projpath, "men0.db")
    colmap_sparse_path = os.path.join(projpath, "sparse")
    makedir(colmap_sparse_path)

    mapper = "colmap mapper --database_path %s --image_path %s --output_path %s" % (
        database_path, images_path, colmap_sparse_path
    )
    operate(mapper)

def geometric_verification(database_path, match_list_path):
    print("Running geometric verification..................................")
    cmd = "colmap matches_importer --database_path %s --match_list_path %s --match_type pairs" % (
        database_path, match_list_path
    )
    operate(cmd)



if __name__  == '__main__':

    data_path="/home/xxl/桌面/men"
    db_path=os.path.join(data_path,"men0.db")
    img_path=os.path.join(data_path,"imagetest")
    if os.path.exists(db_path):
        cmd = "rm -rf %s" % db_path
        operate(cmd)
    db=COLMAPDatabase.connect(db_path)
    db.create_tables()#利用lameda表达式创建
    modelname='SIMPLE_PINHOLE'
    img_dict,id_dict=init_db_camera(db,img_path,modelname,True)
    match_info_dict=add_features(data_path,img_dict,db)
    img_pair_id=image_to_pair_id(img_dict)
    pair_txt="image_pairs_to_match.txt"
    match_dict=get_matches(db,match_info_dict,img_pair_id,pair_txt,id_dict)
    db.commit()
    db.close()
    geometric_verification(db_path,pair_txt)
    mapper(data_path,img_path)
