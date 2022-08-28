# -*- coding: utf-8 -*-
"""
  @Author: zzn 
  @Date: 2019-11-11 19:20:06 
  @Last Modified by:   zzn 
  @Last Modified time: 2019-11-11 19:20:06 
"""
import os
import json
import torch
from PIL import Image, ImageDraw, ImageFont

from model import fasterrcnn_resnet_fpn
from transforms import get_transforms



class Predictor(object):
    def __init__(self, weights_path=None, backbone='resnet50', device='cuda'):
        self.weights_path = weights_path
        if self.weights_path is None:
            self.weights_path = 'checkpoints/weights/best_model.pt'
        self.backbone = backbone
        self.device = device
        self.model = fasterrcnn_resnet_fpn(resnet_name=backbone)
        self.model.load_state_dict(torch.load(
            self.weights_path, map_location=torch.device(device)))
        self.model = self.model.to(device)
        self.model.eval()

    def read_img(self, img_path):
        return Image.open(img_path)

    def process_img(self, img):
        transforms = get_transforms(False)
        img = img.convert('RGB')
        img, _ = transforms(img, None)
        x = img.to(self.device)
        return x

    def predict(self, x):
        with torch.no_grad():
            predictions = self.model([x])
            predictions = {k: v.to('cpu').data.numpy()
                           for k, v in predictions[0].items()}
        return predictions

    def display_boxes(self, img, predictions, img_path, score_thresh=0.75):
        add_img = Image.open(img_path)
        add_img = add_img.convert("RGBA")
        #predictions is a dict,try to show it with elements.
        #print(predictions)
        #print(predictions['boxes'],"-----", predictions['scores'])
        #with open("pos.json",'a+',encoding="utf-8") as f:
        #    f.write(json.dumps({"a":str(predictions['boxes']),"b":str(predictions['scores'])})+"\n")
        #end
        boxes, scores = predictions['boxes'], predictions['scores']
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('data/font.ttf', size=20)
        #for i, cur_bbox in enumerate(boxes):
        if len(boxes) ==0:
            return img
        i=-1
        while i<len(boxes)-1:
            i+=1
            print(i)
            cur_bbox=boxes[i]
            if scores[i] < score_thresh:
                continue
            add_img =add_img.resize((int(cur_bbox[2]-cur_bbox[0]),int(cur_bbox[3]-cur_bbox[1])))
            img.paste(add_img,(int(cur_bbox[0]),int(cur_bbox[1]),int(cur_bbox[0])+add_img.width,int(cur_bbox[1])+add_img.height))
            #draw.rectangle(cur_bbox, outline=(0, 255, 0), width=4)
            #left_corner = (cur_bbox[0]+4, cur_bbox[1]+4)
            #left_corner = (0, i*20)
            #draw.text(left_corner, 'score: {:.4f}'.format(
            #    scores[i]), fill='red', font=font)
        return img
