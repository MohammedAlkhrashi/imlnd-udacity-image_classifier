# Intro to Machine Learning - TensorFlow Project

Project code for Udacity's Intro to Machine Learning with TensorFlow Nanodegree program. 

The classifer was trained on this data http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html featuring 102 category of flowers. 

You can check the notebook for how the model was trained. 

You can use predict.py in order to predict your own flower image,

## Usage: 

use:
```
python predict.py path/to/image path/to/model 
``` 
This will return the top likely classe using the given model,

### Example:
```
python predict.py ./test_images/wild_pansy.jpg best_model.h5
``` 
### Output:
```
1) Class: 51, Probability: 99.95926022529602%
``` 

### Optional Parameter:
#### --top_k returns: the top k classes with their probabilties. 
#### --category_names: a json file that maps each class number with a class name

### Example: 
```
python predict.py ./test_images/orange_dahlia.jpg best_model.h5 --top_k 6 --category_names label_map.json
``` 
### Output:
```
1) Class: orange dahlia, Probability: 59.67817306518555%
2) Class: barbeton daisy, Probability: 16.848114132881165%
3) Class: english marigold, Probability: 11.089663952589035%
4) Class: osteospermum, Probability: 3.103282116353512%
5) Class: gazania, Probability: 2.3248011246323586%
6) Class: black-eyed susan, Probability: 2.2095663473010063%
```

