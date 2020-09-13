from PIL import Image,ImageChops
import glob
import os

way='train'
image_folder = '/disk011/usrs/ogata/Mask_RCNN/data/fastgrape/grape/{}'.format(way)
list =[]
list=glob.glob('{}/*/*_rgb.JPG'.format(image_folder))
i=1
f = open('{}_list.txt'.format(way),'w')
for x in list:
	im= Image.open(x)
	height,width = im.size
	f.write("{} {} {} {}\n".format(i,x,width,height))
	i=i+1
f.close()
