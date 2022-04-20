# Pedestrian-Detection-With-Age-Classification-Pipeline


## Minimum Requiremnts
* Linux OS (Ubuntu)
* 16GB of RAM
* Nvidia GPU with 4GB or more VRAM

## Instructions 

1. ### For enverimentment setup please follow instracunions from [here](https://github.com/hasanirtiza/PedesFormer-Transformer-Networks-For-Pedestrian-Detection)

2. ### Downlaod the model and config files from [here](https://github.com/hasanirtiza/PedesFormer-Transformer-Networks-For-Pedestrian-Detection/tree/main/configs)
3. ### Download pretrained model from [here](http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth) and place in model folder
4.  ### Dowlaod pretrained model from [here](https://drive.google.com/file/d/15paMK0-rKDsuzptDPK5kH2JuL8QO0HyS/view) and place in main directory of repository 
5.  ### Running the pipeline
6.  #### In the terminal run: python detect.py --i "path to image" 


## Refernces:

* Pedestrain Detection Code from [[1]](https://github.com/hasanirtiza/PedesFormer-Transformer-Networks-For-Pedestrian-Detection) [[2]](https://mmdetection.readthedocs.io/en/latest/tutorials/)


* Pedestrain Attribute Classification Code from [[1]](https://github.com/chufengt/iccv19_attribute)


