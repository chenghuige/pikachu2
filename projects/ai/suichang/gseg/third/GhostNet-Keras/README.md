# GhostNet-Keras
Simple implementation 2020 CVPR:《GhostNet：More Features from Cheap Operations》

# 1.Environment:

    keras=2.1.5
    tensorflow=1.4.0
    numpy
    bunch
    opencv
  
# 2.Train mnist data：

    set "use_mnist_data"=1 from config/ghost_config.json, then run command: 
	python train.py -c config/ghost_config.json
  
# 3.Train customer data:
  
    set "use_mnist_data"=0 , "train_list" = path/to/train.txt, "test_list" = path/to/test.txt
    train.txt and test.txt have the following format:
		
    line1: path/to/image1.jpg label(0,1,2,3,4,......)
	line2: path/to/image2.jpg label(0,1,2,3,4,......)
	line3: path/to/image3.jpg label(0,1,2,3,4,......)
	line4: path/to/image4.jpg label(0,1,2,3,4,......)
	....

    run commmand:python train.py -c config/ghost_config.json
    
# 4.Test
    run commmand:python infer.py -c config/ghost_config.json
