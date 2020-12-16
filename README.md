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
* Google Coral

## Part 1: Mask Detection Algorithm

### Training Tool
I use [Google Object Detection API](https://github.com/tensorflow/models) for training deep neural network. It provides more than 20 different models including SSD, Faster RCNN, Mask RCNN to choose except YOLO. Also, most of the models are trained under COCO dataset and evaluated under the same environment ([speed vs accuracy](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)), so we can have better understanding of the performance of models.


### Algorithm Selection
* Fast: need to be real-time detection on raspberry pi so that the processor can have time handle other tasks
* Small Size: able to run on raspberry pi since we only have 4GB RAM
* Tensorflow Lite Compatible: a lightweight library designs for edge devices to deploy models
After evaluating models under this three criterias, I select ssd mobilenet v2 quntized model for my Mask Detection Algorithm.


## Part 2: Body Temperature Sensing

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
