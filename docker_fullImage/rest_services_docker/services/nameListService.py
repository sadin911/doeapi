import pytesseract
import os
import cv2
import pandas as pd
import numpy
class NameListService:
    def __init__(self):
        print('init')

    def nameList(self,img):
        configs = "--psm 1 --oem 3 -l eng"
        img = numpy.array(img) 
        height, width, channels = img.shape 

        rah = [3,1.5,7.35,1.26]
        rav = [2.9,1.4,7.31,1.25]

        if(height<width):
            img2 = img[int(height/rah[0]):int(height/rah[1]) , int(width/rah[2]):int(width/rah[3])]
        else:
            img2 = img[int(height/rav[0]):int(height/rav[1]) , int(width/rav[2]):int(width/rav[3])]
            
        d = pytesseract.image_to_data(img2, config=configs ,output_type=pytesseract.Output.DICT)
        d = pd.DataFrame.from_dict(d)
        d = d[(d['conf'].astype(int)>0)]


        n_boxes = len(d['conf'])
        indexs = d.index
        indexed = []
        linenum_prev = 0
        lineText = ''
        lineList = []
        nameList = []
        passportList = []

        for i in range(len(indexs)):
            j = indexs[i].astype(int)
            if((''.join(set(d['text'][j])))!=" " and ''.join(set(d['text'][j]))!="|" and ''.join(set(d['text'][j]))!=""):
                linenum = str(d['line_num'][j])
                if(linenum==linenum_prev):
                    lineText = lineText + d['text'][j] + ' '
                    linenum_prev = linenum
                else:
                    print(lineText)
                    lineList.append(lineText)
                    lineText = d['text'][j] + ' '
                    linenum_prev = linenum
            
                
                (x, y, w, h) = (d['left'][j], d['top'][j], d['width'][j], d['height'][j])
                cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img2,d['text'][j], (x,y-7), 2, 0.7, (255,0,255))
                cv2.putText(img2,str(d['line_num'][j]), (x-15,y-7), 2, 0.7, (255,0,0))
                indexed.append(j)
                
        lineList.append(lineText)
        for i in range(len(lineList)):
            if(lineList[i]) != '' :
                linesplit = lineList[i].split(' ')
                passport = linesplit[-2]
                name = linesplit[:-4]
                passportList.append(passport)
                nameList.append(name)
        
        return(nameList,passportList)

    def nameListCropped(self,img):
        configs = "--psm 1 --oem 3 -l eng"
        img2 = numpy.array(img) 

        d = pytesseract.image_to_data(img2, config=configs ,output_type=pytesseract.Output.DICT)
        d = pd.DataFrame.from_dict(d)
        d = d[(d['conf'].astype(int)>0)]


        n_boxes = len(d['conf'])
        indexs = d.index
        indexed = []
        linenum_prev = 0
        lineText = ''
        lineList = []
        nameList = []
        passportList = []

        for i in range(len(indexs)):
            j = indexs[i].astype(int)
            if((''.join(set(d['text'][j])))!=" " and ''.join(set(d['text'][j]))!="|" and ''.join(set(d['text'][j]))!=""):
                linenum = str(d['line_num'][j])
                if(linenum==linenum_prev):
                    lineText = lineText + d['text'][j] + ' '
                    linenum_prev = linenum
                else:
                    print(lineText)
                    lineList.append(lineText)
                    lineText = d['text'][j] + ' '
                    linenum_prev = linenum
            
                
                (x, y, w, h) = (d['left'][j], d['top'][j], d['width'][j], d['height'][j])
                cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img2,d['text'][j], (x,y-7), 2, 0.7, (255,0,255))
                cv2.putText(img2,str(d['line_num'][j]), (x-15,y-7), 2, 0.7, (255,0,0))
                indexed.append(j)
                
        lineList.append(lineText)
        for i in range(len(lineList)):
            if(lineList[i]) != '' :
                linesplit = lineList[i].split(' ')
                passport = linesplit[-2]
                name = linesplit[:-4]
                passportList.append(passport)
                nameList.append(name)
        
        return(nameList,passportList)