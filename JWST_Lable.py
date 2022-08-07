# -*- coding: utf-8 -*-
"""
@Time ： 2022/7/26 9:01
@Auth ： zyt_sky
@File ：JWST_Lable.py
@IDE ：PyCharm
@Email: a2534487689@qq.com
@Motto：大威天龙 大罗法咒
"""
from astropy.io import fits
import pandas as pd
import shutil
import glob
import json
import csv
import os


def txt2csv(txt_name, csv_name):
    csv_file = open(csv_name, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    csv_row = []

    f = open(txt_name, 'r', encoding='GB2312')
    for line in f:
        csv_row = line.split()
        writer.writerow(csv_row)

    f.close()
    csv_file.close()


def generate_csv(cat_list):
    for cat in cat_list:
        with open(cat, 'r') as f:
            text = f.readlines()

        del text[0:8]
        with open(cat.replace('.cat', '.txt'), 'a+') as f:
            pa_str = 'X_IMAGE  Y_IMAGE  A_IMAGE  B_IMAGE THETA_IMAGE CLASS_STAR MAG_AUTO MAG_BEST\n'
            f.write(pa_str)
            f.writelines(text)
        txt2csv(cat.replace('.cat', '.txt'), cat.replace('.cat', '.csv'))


def call_gen_csv():
    cat_list = glob.glob(r'E:\JWST\ge\*.cat')
    generate_csv(cat_list)


def remove_file(file_list):
    for file in file_list:
        os.remove(file)


def remove_all():
    remove_file(glob.glob(r'E:\JWST\star\train\*.fits'))
    remove_file(glob.glob(r'E:\JWST\star\train\*.json'))
    remove_file(glob.glob(r'E:\JWST\star\val\*.fits'))
    remove_file(glob.glob(r'E:\JWST\star\val\*.json'))
    remove_file(glob.glob(r'E:\JWST\galaxy\train\*.fits'))
    remove_file(glob.glob(r'E:\JWST\galaxy\val\*.json'))
    remove_file(glob.glob(r'E:\JWST\galaxy\train\*.fits'))
    remove_file(glob.glob(r'E:\JWST\galaxy\val\*.json'))
    print('remove all done')


def remove_swin():
    remove_file(glob.glob(
        r'E:\Galaxy_detection\Swin-Transformer-Object-Detection-master\data\coco\train2017_JWST_galaxy\*.fits'))
    remove_file(glob.glob(
        r'E:\Galaxy_detection\Swin-Transformer-Object-Detection-master\data\coco\train2017_JWST_galaxy\*.json'))
    remove_file(
        glob.glob(r'E:\Galaxy_detection\Swin-Transformer-Object-Detection-master\data\coco\val2017_JWST_galaxy\*.fits'))
    remove_file(
        glob.glob(r'E:\Galaxy_detection\Swin-Transformer-Object-Detection-master\data\coco\val2017_JWST_galaxy\*.json'))

    remove_file(
        glob.glob(r'E:\Galaxy_detection\Swin-Transformer-Object-Detection-master\data\coco\train2017_JWST_star\*.fits'))
    remove_file(
        glob.glob(r'E:\Galaxy_detection\Swin-Transformer-Object-Detection-master\data\coco\train2017_JWST_star\*.json'))
    remove_file(
        glob.glob(r'E:\Galaxy_detection\Swin-Transformer-Object-Detection-master\data\coco\val2017_JWST_star\*.fits'))
    remove_file(
        glob.glob(r'E:\Galaxy_detection\Swin-Transformer-Object-Detection-master\data\coco\val2017_JWST_star\*.json'))
    remove_file(
        glob.glob(r'E:\Galaxy_detection\Swin-Transformer-Object-Detection-master\data\coco\annotations_JWST_galaxy'
                  r'\*.json'))
    remove_file(
        glob.glob(r'E:\Galaxy_detection\Swin-Transformer-Object-Detection-master\data\coco\annotations_JWST_star'
                  r'\*.json'))
    print('remove swin done')


def remove_ge():
    remove_file(glob.glob(r'E:\JWST\ge\*.txt'))
    remove_file(glob.glob(r'E:\JWST\ge\*.csv'))
    remove_file(glob.glob(r'E:\JWST\ge\*.json'))
    print('remove ge done')


def gen_star_json(csv_list):
    print('start generate star json .....')
    for csv_file in csv_list:
        data = fits.getdata(csv_file.replace('.csv', '.fits'))
        h, w = data.shape[0], data.shape[1]
        df = pd.read_csv(csv_file)

        df = df.loc[(df['MAG_AUTO'] < 0)].reset_index()
        star = df.loc[(df['CLASS_STAR'] == 0)].reset_index()
        cx = star['X_IMAGE']
        cy = star['Y_IMAGE']
        bc = star['A_IMAGE']
        bd = star['B_IMAGE']

        new_shapes = []
        for j in range(len(cx)):
            left_top_x = cx[j] - (bc[j] / 2)
            left_top_y = cy[j] - (bd[j] / 2)
            right_bottom_x = cx[j] + (bc[j] / 2)
            right_bottom_y = cy[j] + (bd[j] / 2)

            if left_top_x < h and left_top_y < w and right_bottom_x < h and right_bottom_y < w and left_top_x > 0 and left_top_y > 0 and right_bottom_x > 0 and right_bottom_y > 0:
                points = [[left_top_x, left_top_y], [right_bottom_x, right_bottom_y]]
                info = {
                    "label": "star",
                    "points": points,
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
                new_shapes.append(info)
            all_info = {
                "version": "4.5.13",
                "flags": {},
                "shapes": new_shapes,
                "imagePath": csv_file.replace('.csv', '.fits').split('\\')[-1],
                "imageData": "wo fang ni zai zhe li yi si yi si ",
                "imageHeight": h,
                "imageWidth": w,

            }
            with open(csv_file.replace('.csv', '_star.json'), 'w', encoding='utf-8') as fw:
                json.dump(all_info, fw, indent=4, ensure_ascii=False)
    print('done')


def gen_galaxy_json(csv_list):
    print('start generate galaxy json .....')
    for csv_file in csv_list:

        data = fits.getdata(csv_file.replace('.csv', '.fits'))
        h, w = data.shape[0], data.shape[1]
        df = pd.read_csv(csv_file)

        df = df.loc[(df['MAG_AUTO'] < 0)].reset_index()
        galaxy = df.loc[(df['CLASS_STAR'] > 0.005)].reset_index()
        cx, cy = galaxy['X_IMAGE'], galaxy['Y_IMAGE']
        bd = galaxy['B_IMAGE']

        shapes = []
        for i in range(len(cx)):
            points = [[cx[i], cy[i] - bd[i]], [cx[i], cy[i] + bd[i]], [cx[i] + bd[i], cy[i]], [cx[i] - bd[i], cy[i]]]
            info = {
                "label": "galaxy",
                "points": points,
                "group_id": None,
                "shape_type": "polygons",
                "flags": {}
            }
            shapes.append(info)

        all_info = {
            "version": "4.5.13",
            "flags": {},
            "shapes": shapes,
            "imagePath": csv_file.replace('.csv', '.fits').split('\\')[-1],
            "imageData": "wo fang ni zai zhe li yi si yi si ",
            "imageHeight": h,
            "imageWidth": w,

        }
        with open(csv_file.replace('.csv', '_galaxy.json'), 'w', encoding='utf-8') as fw:
            json.dump(all_info, fw, indent=4, ensure_ascii=False)
    print('done')


def create_dataset():
    print('create swin dataset ...')
    star_json = glob.glob(r'E:\JWST\ge\*_star.json')
    galaxy_json = glob.glob(r'E:\JWST\ge\*_galaxy.json')
    train_star_dst = r'E:\Galaxy_detection\Swin-Transformer-Object-Detection-master\data\coco\train2017_JWST_star'
    val_star_dst = r'E:\Galaxy_detection\Swin-Transformer-Object-Detection-master\data\coco\val2017_JWST_star'
    train_galaxy_dst = r'E:\Galaxy_detection\Swin-Transformer-Object-Detection-master\data\coco\train2017_JWST_galaxy'
    val_galaxy_dst = r'E:\Galaxy_detection\Swin-Transformer-Object-Detection-master\data\coco\val2017_JWST_galaxy'
    for sj in star_json[0:1200]:
        base = sj.replace('_star.json', '.fits').split("\\")[-1]
        new_train_star_dst = os.path.join(train_star_dst, base)
        print(new_train_star_dst)
        shutil.move(sj, train_star_dst)
        shutil.copyfile(sj.replace('_star.json', '.fits'), new_train_star_dst)
    for sj in star_json[1200:1500]:
        base = sj.replace('_star.json', '.fits').split("\\")[-1]
        new_val_star_dst = os.path.join(val_star_dst, base)
        print(new_val_star_dst)
        shutil.move(sj, val_star_dst)
        shutil.copyfile(sj.replace('_star.json', '.fits'), new_val_star_dst)
    for gj in galaxy_json[0:1200]:
        base = gj.replace('_galaxy.json', '.fits').split("\\")[-1]
        new_train_galaxy_dst = os.path.join(train_galaxy_dst, base)
        print(new_train_galaxy_dst)
        shutil.move(gj, train_galaxy_dst)
        shutil.copyfile(gj.replace('_galaxy.json', '.fits'), new_train_galaxy_dst)
    for gj in galaxy_json[1200:1500]:
        base = gj.replace('_galaxy.json', '.fits').split("\\")[-1]
        new_val_galaxy_dst = os.path.join(val_galaxy_dst, base)
        print(new_val_galaxy_dst)
        shutil.move(gj, val_galaxy_dst)
        shutil.copyfile(gj.replace('_galaxy.json', '.fits'), new_val_galaxy_dst)
    print('done')


def get_all_json():
    csv_list = glob.glob(r'E:\JWST\ge\*.csv')
    gen_star_json(csv_list)
    gen_galaxy_json(csv_list)


if __name__ == '__main__':
    remove_ge()
    remove_all()
    remove_swin()
    # call_gen_csv()
    # get_all_json()
    # create_dataset()
