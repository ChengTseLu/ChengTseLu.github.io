import argparse
import cv2
from TFLite import tflite
from mlx90614 import thermal_sensor
from cloud import cloud
from GUI import gui

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                        default='mask_detection')
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                        default='mask.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.5)
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='600x600')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')
    args = parser.parse_args()

    # initialize classes
    Mask = tflite(args.modeldir, args.graph, args.labels, args.threshold, args.resolution, args.edgetpu)
    Temp = thermal_sensor()
    GUI = gui()
    cloud = cloud()
    
    # initialize variables
    detect = []
    counter = 0
    max_f = 30 if args.edgetpu else 4       # using google coral has better fps, so need to increase num of frame count
    max_c = 120 if args.edgetpu else 16     # using google coral has better fps, so need to increase num of frame count

    while True:
        # get mask detection result and temperature
        frame, result = Mask.get_frame()
        sur_temp, obj_temp = Temp.temp()

        # store mask detection into array (max size: 30 frames using google coral)
        detect.append(result)
        if len(detect) > max_f:
            detect.pop(0)

        # averaging the detection within 30 frames to
        # avoid bad detection results
        # don't count temperature lower than 30 
        if detect.count("Mask") > max_f/2:
            mask_check = "Mask"
        elif detect.count(None) < max_f/2:
            mask_check = "Nomask"
        else:
            mask_check = None
            
        obj_temp = None if obj_temp < 30 else round(obj_temp,2)
        
        # set text to GUI
        GUI.setTxt(frame, mask_check, obj_temp)
        
        # save the mask detection result to cloud database under these conditions
        if mask_check:
            counter += 1
            if counter > max_c and obj_temp:
                counter = 0
                cloud.service(mask_check, obj_temp)
        else:
            counter = 0
        
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            Mask.stop()
            cv2.destroyAllWindows()
            break