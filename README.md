# Mask Detection with Body Temperature Sensing

## Team Member
* Cheng-Tse Lu

## Project Introduction
**Motivation**  

Due to COVID-19 pandemic, there are more than 40 million confirmed cases, with 1 million people deaths attributed to the pandemic. COVID-19 spreads very easily through the air when an infected person or an asymptomatic infection breathes, coughs, talks, or sneezes. Centers for Disease Control and Prevention (CDC) recommends that “people wear masks in public settings, like on public and mass transportation, at events and gatherings, and anywhere they will be around other people” and “masks may help prevent people who have COVID-19 from spreading the virus to others.” Therefore, having a system that can detect whether a person is wearing a mask and also measuring his/her body temperature at the same time could keep tracking people's health; furthermore, reduce the risk of being infected.

**Goal**  

Incorporate deep learning algorithm for mask detection and use the sensor to measure body temperature. Also, sort the data in cloud.

**Deliverable**  

Instead of using thermography camera for long range and multiple detections, I use small thermal sensor with edge AI computing to achieve the goal.

## Hardware Used in This Project
* Raspberry Pi 4 Model B (4GB RAM)
* Raspberry Pi camera v2.1
* GY-MLX90614-DCI
* Google Coral (Edge TPU)

## Installation on Raspberry Pi
* Download the github project

        git clone https://github.com/ChengTseLu/MaskDetectionTempSensing.git
        cd MaskDetectionTempSensing
        
* Install python virtual environment

        sudo apt-get install python-virtualenv
        pip3 install virtualenv
        
* Create python virtual environment
        
        python3 -m venv project-env
        
* Activate python virtual environment (activate everytime when open new terminal)

        source project-env/bin/activate
        
* Install packages
        
        bash get_pi_requirements.sh
        pip3 install adafruit-circuitpython-mlx90614
        pip3 install requests
        pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37m-linux_armv7l.whl
        
* Install Google Coral Library

        echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
        curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
        sudo apt-get update
        sudo apt-get install libedgetpu1-std

* If error indicates no module named 'cv2', delete the project-env folder, deactivate the virtual environment and reinstall the packages again (try couple times)


## Part 1: Mask Detection Algorithm

### Training Tool
I use [Google Object Detection API](https://github.com/tensorflow/models) for training deep neural network. It provides more than 20 different models including SSD, Faster RCNN, Mask RCNN to choose except YOLO. Also, most of the models are trained under COCO dataset and evaluated under the same environment ([speed vs accuracy](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)), so we can have better understanding of the performance of models.


### Algorithm Selection
* Fast: need to be real-time detection on raspberry pi so that the processor can have time handle other tasks
* Small Size: able to run on raspberry pi since we only have 4GB RAM
* Tensorflow Lite Compatible: a lightweight library designs for edge devices to deploy models (Note: Tensorflow Lite does not support RCNN models, only SSD models) 
  
After evaluating models under this three criterias, I select ssd mobilenet v2 quantized model for my Mask Detection Algorithm (quantizing the model from FP32 (float) to INT8 (int) increase the speed and reduce the model size; however, the accuracy would also decrease)

### Dataset
I use an open source [dataset](https://github.com/AIZOOTech/FaceMaskDetection) which contain 7959 images and bounding boxes of both mask and nomask. According to their description, "the dataset is composed of WIDER Face and MAFA, and we verified some wrong annotations." The file can be downloaded through [google drive](https://drive.google.com/file/d/1QspxOJMDf_rAWVV7AU_Nc0rjo1_EPEDW/view) (763Mb)

### Training Steps 
* Step 1: Generating TFRecords from dataset
* Step 2: Configuring training
* Step 3: Training model
* Step 4: Freeze model to .tflite
* Step 5: Convert .tflite to Edge TPU compatible model
  
More details about training steps can be found online ([offical doc](https://towardsdatascience.com/creating-your-own-object-detector-ad69dda69c85))  
More details about freezing model to .tflite ([offical doc](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md))  
More details about converting to edge tpu compatible model ([offical doc](https://coral.ai/docs/edgetpu/compiler/#system-requirements))

### Raspberry Pi Implement
The mask detection model is trained and uploaded in the mask_detection folder  
Run a simple test using following command (make sure Raspberry Pi camera is installed)

        cd MaskDetectionTempSensing
        python3 TFlite.py
        

## Part 2: Body Temperature Sensing
[GY-MLX90614-DCI](https://www.digchip.com/datasheets/parts/datasheet/287/MLX90614ESF-DCI-000-SP-pdf.php) has a high precision contactless temperature measurement. The sensor can detect object temperature from -70 to +380 C range with a 0.02 C resolution.  
Raspberry Pi GPIO Connection
![](https://github.com/ChengTseLu/MaskDetectionTempSensing/blob/main/pic/mlx90614.png)

## Part 3: Cloud Storage

## Final: Combine 3 Parts
**Demo**

## Summary
**Strengths**

**Weakness**

**Future Directions**

**Final Presentation Slide**

## References
**Data Sets**
