#encoding=utf8
'''
Detection with SSD
In this example, we will load a SSD model and use it to detect objects.
'''

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw
# Make sure that caffe is on the python path:
caffe_root = os.getcwd()
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))

import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in range(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames
    print (labelnames)
class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights, image_resize, labelmap_file):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()
        self.image_resize = image_resize
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
        #self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        #self.transformer.set_transpose('data', (2, 0, 1)) ############
        #self.transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        #self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        #self.transformer.set_channel_swap('data', (2, 1, 0))
        # load PASCAL VOC labels


        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

    def detect(self, image_file, conf_thresh=0.5, topn=5):
        '''
        SSD detection
        '''
        # set net to batch size of 1
        # image_resize = 300

        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)
        image = cv2.imread(image_file)
        image = cv2.resize(image,(300,300))
        image = image -127.5
        image = image *0.007843
        image = image.astype(np.float32)
        transformed_image = image.transpose((2,0,1))


        #Run the net and examine the top_k results
        #transformed_image = self.transformer.preprocess('data', image)

        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = self.net.forward()['detection_out']#############  out
        print('detections',detections)
        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]
        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        result1=[]
        for i in range(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            display_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, label, score, display_name])
            result1.append([display_name,score,xmin,ymin,xmax,ymax])
            #print (str(result1))
            #print ('aaaaaaaaaaaaaaaaaaaaaaaaaa')
            #f= open('/data0/workspace/limei/data/predict/'+str(jj).rstrip('.jpg')+'.txt','w')
            #f.write(str(result1).strip('[').replace("'","").replace(',',' ').strip(']'))

            #f.write('\n')
            #f.close()
        return result,result1

def main(args):
    '''main '''
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file)
    file_dir =os.listdir('/data0/workspace/limei/code/mAP/images/')

    for j in file_dir:
        
        #print (file_dir[i])

        file_output = os.path.join('/data0/workspace/limei/data/19c_v1_result/'+str(j))
        print (file_output)
        file_images=os.path.join('/data0/workspace/limei/code/mAP/images/'+str(j))
        result,result1 = detection.detect(file_images)
    
        img = Image.open(file_images)
        draw = ImageDraw.Draw(img)
        width,height=img.size
        for ii in result1:
                ii[2]=int(round(ii[2]*width))
                ii[3]=int(round(ii[3]*height))
                ii[4]=int(round(ii[4]*width))
                ii[5]=int(round(ii[5]*height))
      

        f=open('/data0/workspace/limei/data/19c_v1_result/predicted/'+str(j).strip('.jpg')+'.txt','w')
        print('result1=',result1)
        #print('result=',result)
        if (len(result1)<2):
            f.write(str(result1).replace(']','').replace('[','').replace(',',' ').replace("'",""))
            #f.write('\n')
            #f.write(str(result1[0][6:]).replace(']','').replace('[','').replace(',',' ').replace("'",""))
            
        else:
            f.write(str(result1[0]).replace(']','').replace('[','').replace(',',' ').replace("'",""))
            f.write('\n')
            f.write(str(result1[1]).replace(']','').replace('[','').replace(',',' ').replace("'",""))
            
        f.close()

        #img = Image.open(file_images)
        #draw = ImageDraw.Draw(img)
        #width, height = img.size
        #print('width=',width,'height=', height)
        for item in result:
            xmin = int(round(item[0] * width))
            ymin = int(round(item[1] * height))
            xmax = int(round(item[2] * width))
            ymax = int(round(item[3] * height))
            draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))
            draw.text([xmin, ymin], item[-1] + str(item[-2]), (0, 0, 255))
      # print item
                # print [xmin, ymin, xmax, ymax]
                # print [xmin, ymin], item[-1]
        img.save(file_output)
        

def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default='1', help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='/data0/workspace/limei/code/trainfiles/labelmap_voc.prototxt')
    parser.add_argument('--model_def',
                        default='/data0/workspace/wuxianli/Mobilenet/code/caffe/examples/MobileNet-SSD-master/example/MobileNetSSD_deploy.prototxt')
    parser.add_argument('--image_resize', default=300, type=int)
    parser.add_argument('--model_weights',
                        default='/data0/workspace/wuxianli/Mobilenet/code/caffe/examples/MobileNet-SSD-master/MobileNetSSD_deploy_my_test_1000.caffemodel')
    parser.add_argument('--image_file', default='/data0/workspace/limei/code/mAP/images/')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
