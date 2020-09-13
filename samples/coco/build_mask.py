#coding : utf-8
from __future__ import unicode_literals, print_function, absolute_import
from PIL import Image,ImageChops
import numpy as np
# from scipy import misc
import cv2
import sys
import glob
import os

# list=[]
# list=glob.glob('finish_confirmation/*label.png')
# #listにラベル画像を取得、ラベル画像は背景が黒で対象物は黒と白以外で塗ってあれば問題無し

def Build_mask(image_name):
	cnt=0#ブドウの実が異常な数に達した際に脱出する用
	instance_masks=[]
	class_ids = []
	im = Image.open(image_name).convert("RGB")#画像を開く
# 	name,ext=os.path.splitext(x)#取得した画像名を名前と拡張子に分離
	cp = im.copy()#元画像を残したいので、画像をコピー
	r,g,b=cp.split()#画像をrgbとなんかに分ける
	height,width = cp.size#サイズを取得
	#次で全画素探索、ここが遅すぎるから早くしたい
	for i in range(int(height)):
		for j in range(int(width)):
			_red,_green,_blue=cp.getpixel((i,j))#コピーの１画素ずつをrgbで取得
			if _red!=0 or _green!=0 or _blue!=0:#全画素値において、黒以外の部分を検索
				src_color = cp.getpixel((i,j))
				_r = r.point(lambda _:1 if _ == src_color[0] else 0, mode="1")
				_g = g.point(lambda _: 1 if _ == src_color[1] else 0, mode="1")
				_b = b.point(lambda _: 1 if _ == src_color[2] else 0, mode="1")
				mask = ImageChops.logical_and(_r,_g)#マスク画像を生成、
				mask = ImageChops.logical_and(mask,_b)#，，，
				m=np.asarray(mask,dtype=np.int32)
				cp.paste(Image.new("RGB",cp.size,(0,0,0)),mask=mask)
				if m.max()<1:
					break
				if _red==255 and _green==255 and _blue==255:
					class_id=1
					instance_masks.append(m)
					class_ids.append(class_id)
				else:
					class_id=2
					instance_masks.append(m)
					class_ids.append(class_id)
	return instance_masks,class_ids
