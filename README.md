# Pedestrian-Detection-With-Age-Classification-Pipeline


## Minimum Requirements
* Linux OS (Ubuntu)
* 16GB of RAM
* Nvidia GPU with 4GB or more VRAM

## Instructions 

1. ### For environment setup please follow instructions from [here](https://github.com/hasanirtiza/PedesFormer-Transformer-Networks-For-Pedestrian-Detection)

2. ### Download the model and config files from [here](https://github.com/hasanirtiza/PedesFormer-Transformer-Networks-For-Pedestrian-Detection/tree/main/configs)
3. ### Download pretrained model from [here](http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth) and place in model folder
4.  ### Download pretrained model from [here](https://drive.google.com/file/d/15paMK0-rKDsuzptDPK5kH2JuL8QO0HyS/view) and place in main directory of repository 
5.  ### Running the pipeline
6.  #### In the terminal run: python detect.py --i "path to image" 


## Note
For training the models, I have used code from [[here]](https://github.com/hasanirtiza/PedesFormer-Transformer-Networks-For-Pedestrian-Detection). To train the models further:
1. Cone that repository
2. Dwnload cityscapes dataset from [here](https://www.cityscapes-dataset.com/login/)
3. Use the configuration files
4. Then run 
```
Python tools/train.py config_file_path –resume-from model_file_path
```
Please check the [instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for information

## References:

* Pedestrian Detection Code from [[1]](https://github.com/hasanirtiza/PedesFormer-Transformer-Networks-For-Pedestrian-Detection) [[2]](https://mmdetection.readthedocs.io/en/latest/tutorials/)


* Pedestrian Attribute Classification Code from [[1]](https://github.com/chufengt/iccv19_attribute)


