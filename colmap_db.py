from sqlite3.dbapi2 import Cursor
import numpy as np
import sqlite3
import cv2
import os
import itertools
import matplotlib.pyplot as plt

def init_database(paths):
    path=os.path.join(paths,"men.db")
    conn=sqlite3.connect(path)
    cursor=conn.cursor()

    sql="DELETE from two_view_geometries"
    cursor.execute(sql)
    sql0="DELETE from descriptors"
    cursor.execute(sql0)
    sql1="DELETE from matches"
    cursor.execute(sql1)
    sq2="DELETE from keypoints"
    cursor.execute(sq2)
    conn.commit()
    cursor.close()
    conn.close()
    return 0

def import_db(images_id,paths):
    path=os.path.join(paths,"men.db")
    conn=sqlite3.connect(path)
    cursor=conn.cursor()

    des_list=[]
    for idx,image in images_id.items():
        image_path=os.path.join(paths+"/imagetest",image)
        img=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        # cv2.imshow("window",img)
        # cv2.waitKey()
        sift=cv2.SIFT_create(12000)
        # orb=cv2.ORB_create()
        kp_s,des_s=sift.detectAndCompute(img,None)
        # kp_o,des_o=orb.detectAndCompute(img,None)
        des_list.append(des_s)
        # print(len(kp_s))
        # print(len(kp_o))
        des_s=np.array(des_s)
        kp_list=[]
        for kp in kp_s:
            kp_list.append([kp.pt[0],kp.pt[1],kp.size,kp.angle])#像素坐标，尺度，主方向
            #保留xy坐标
        kp_list=np.array(kp_list).reshape(-1,4)
        kp_list=kp_list[:,:2]
  
        kp_data=np.concatenate([kp_list,np.ones([kp_list.shape[0],1]),np.zeros([kp_list.shape[0],1])],axis=1).astype(np.float32)
        kp_data_str=kp_data.tostring()
        sql="INSERT INTO keypoints (image_id,rows,cols,data)VALUES(?,?,?,?);"
        cursor.execute(sql,(idx,kp_data.shape[0],kp_data.shape[1],kp_data_str))

        des_data_str=des_s.tostring()
        sql1="INSERT INTO descriptors (image_id,rows,cols,data)VALUES(?,?,?,?);"
        cursor.execute(sql1,(idx,des_s.shape[0],des_s.shape[1],des_data_str))

    conn.commit()
    cursor.close()
    conn.close()

    return des_list


def get_matches(path,des_list,pair_id):
    DB_path=os.path.join(path,"men.db")
    conn=sqlite3.connect(DB_path)
    cursor=conn.cursor()
    pair_list=[]
    match_dict={}

    for id in pair_id:
        pair=(id//2147483647,id%2147483647,id)# 列表不能做dict的key
        # pair.append(id//2147483647)
        # pair.append(id%2147483647)
        pair_list.append(pair) 

    
    FLANN_INDEX_KDTREE=0
    indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    searchParams= dict(checks=50)

    for id, pair in enumerate(pair_list):
        des1=des_list[pair[0]-1]
        des2=des_list[pair[1]-1]
        flann=cv2.FlannBasedMatcher(indexParams,searchParams)
        matches=flann.knnMatch(des1,des2,2)
        match_info=[]

        idx=0
        while idx<len(matches):
        # for idx,(m,n) in enumerate(matches):
            #去除错误匹配
            m,n=matches[idx]
            if m.distance>0.8*n.distance:#两个相近的匹配要距离差距足够大才保留
                matches.pop(idx)
               
            else:
                # print(matches[idx][0].distance)
                # print(matches[idx][0].queryIdx)
                # print(matches[idx][0].trainIdx)
                match_info.append((matches[idx][0].distance,matches[idx][0].queryIdx,matches[idx][0].trainIdx))#保留匹配最好的数据
                idx+=1
            
        match_info=np.array(sorted(match_info))[:,1:].astype(np.int16)#按照距离排序，并去除距离信息\

        # pair_idx=list(pair_id.keys())[id]
        pair_idx=pair_list[id]
        match_dict[pair_idx]=match_info#.tolist()
        matches_str=match_info.tostring()

        # pair_idx=2147483647*pair[0]+pair[1]
        # print(list(pair_id.keys())[id])
        cursor.execute("INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);",
                   (list(pair_id.keys())[id], match_info.shape[0], match_info.shape[1], matches_str))
    conn.commit()
    cursor.close()
    conn.close()
    return match_dict


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
    
# def img_id_to_pair(image_id1,image_id2):
#     if image_id1 > image_id2:
#         pair_id=2147483647 * image_id2 + image_id1
#     else:
#         pair_id=2147483647 * image_id1 + image_id2
#     # print(pair_id)
#     return pair_id

def add_two_views_geometry(match_dict_item,db_path,F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2):
    DB_path=os.path.join(db_path,"men.db")

    conn=sqlite3.connect(DB_path)
    cursor=conn.cursor()
  
    key,value=match_dict_item
    assert (value.shape[1]==2)
    if key[0]>key[1]:
        value=value[:,::-1]#交换匹配组内两点索引
    pair_id=key[2]
    value_str=value.tostring()    
    F=np.asarray(F,dtype=np.float64).tostring()
    E=np.asarray(E,dtype=np.float64).tostring()
    H=np.asarray(H,dtype=np.float64).tostring()
    cursor.execute("INSERT INTO two_view_geometries(pair_id, rows, cols, data,config,F,E,H) VALUES(?, ?, ?, ?,?,?,?,?);",
                (pair_id,value.shape[0], value.shape[1], value_str,config,F,E,H))
    conn.commit() 
    cursor.close()
    conn.close()

def draw_match_point(img_id_dict,match_dict,paths):
    key,value=match_dict
    assert(len(key)==3)
    flag1,flag2=(False,False)
    for key,value in img_id_dict.items():
        if flag1&flag2==False:
            if key==match_dict[0][0]:
                img1=cv2.imread(os.path.join(paths,value))
                flag1=True
                print(value)
            if key==match_dict[0][1]:
                img2=cv2.imread(os.path.join(paths,value))
                flag2=True
                print(value)
        else:
            break

   
    sift=cv2.SIFT_create(12000)
    kps1,des1=sift.detectAndCompute(img1,None)
    kps2,des2=sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE=0
    indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    searchParams= dict(checks=50)
    flann=cv2.FlannBasedMatcher(indexParams,searchParams)
    matches=flann.knnMatch(des1,des2,2)
    matchesMask = [[0,0] for i in range(len(matches))]

    print(len(matches))
    count=0
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            matchesMask[i]=[1,0]
            count+=1
    print(count)
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

    img3 = cv2.drawMatchesKnn(img1,kps1,img2,kps2,matches,None,**draw_params)
    plt.imshow(img3)
    plt.show()

def connect_img_id(db_path):
    DB_path=os.path.join(db_path,"men.db")
    conn=sqlite3.connect(DB_path)
    cursor=conn.cursor()
    sql0="SELECT camera_id,name FROM images"

    img_name=cursor.execute(sql0)
    img_id_dict=dict(list(img_name))
    cursor.close()
    conn.close()
    return img_id_dict
    


if __name__=="__main__":
    file_path="/home/xxl/桌面/men/imagetest"
    # img_list=[]
    # for file in os.listdir(file_path):
    #     # abspath=os.path.join(file_path,file)
    #     # img=cv2.imread(abspath)
    #     img_list.append(file)
    # # print(len(img_list))
    path="/home/xxl/桌面/men"
    init_database(path)
    img_id_dict=connect_img_id(path)
    pair_id=image_to_pair_id(img_id_dict)
    des_list=import_db(img_id_dict,path)
    match_dict=get_matches(path,des_list,pair_id)
    for item in match_dict.items():
        add_two_views_geometry(item,path)
    

    print("modify to github")
    # draw_match_point(img_id_dict,list(match_dict.items())[2],file_path)
#     aa = [1, 2,3]
#     bb = list(itertools.permutations(aa, 2))
    # for item in bb:
    #     id=image_ids_to_pair_id(item[0], item[1])
    #     print(id)

