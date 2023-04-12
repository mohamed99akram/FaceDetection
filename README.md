# features_description.py

* Gets descriptions for all features

```python
describe_features(shape)
```
for the prvided shape:   
returns: *indecies* of: 2-rect, 3-rect, 4-rect
2-rect: (positive, negative)
3-rect: (negative, positive, negative)
4-rect: (negative, positive, positive, negative)

```python
getIntegrralImage(img)
```
input: list of gray images | or one image  
returns: list of integral images for each input image | or one integral image  

```python
getFeatureValue(ii,f)
```
**input:**   
- ii: integral image | or list of integral images    
- f : [  
    i1,j1, i2,j2 (first feature)  
    i1,j1, i2,j2 (second feature)  
    ...  
  ]  

**returns:** 
- area of the rectangle for each feature for all given images in ii
  

# feature_extractor.ipynb
- reads the dataset
- extracts each of the 2-rect, 3-rect, 4-rect features separately using Dataloader 
- puts all features in one .npz file of shape:(number of features, number of images)

# accumulative.ipynb  ❤❤
<p>Comparison of original implementation of choosing strong classifier with numpy and pytorch.</p>

### idea
- input: list of features for each image in the training set
- it is of shape: (number of features, number of images)
- each image has a weight
- required to find a threshold for each feature that best separates the positive and negative images
- 'best separates': minimum sum of weights of misclassified images
- for each feature (each row of the input): the best feature will be one of the images
- **if we sort** the row by feature value, one side will be negative and the other positive
- left can be positive or negative, right can be positive or negative, so we need to test polarities
- we need to decide which feature best separates the two sides
- To do so, we have accumulative sum of weights for misclassified images on the left and sum of weights of misclassified images on the right
- the sum of these two is the sum of weights of misclassified images for the current feature
- we find the value of the feature that gives the minimum sum of weights of misclassified images
- this value is considered the threshold for the current feature
- we use pytorch to sort and accumulate the weights and find the minimum on the axis=1 to use GPU and make it faster
- Our weak classifier is the threshold and the polarity. if given a new image, it will compute this feature for the image, compare with the threshold and based on the polarity, will decide whether it is a face or not
- The returned value is the index of the feature (index of the row) that gives overall minimum sum of weights of misclassified images (minimum error) and the threshold for that feature and the polarity

# feat_desc.ipynb  
This is just a test file for the features_description.py, it compares its output with other implementations of the same features