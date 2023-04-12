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
- The returned value is the index of the feature (index of the row) that gives overall minimum sum of weights of misclassified images (minimum error) and the threshold for that feature and the polarity and the error
  

Example:
<!-- add drop down with html, default: open-->

<details open> 
<summary>Example</summary>

```python
f = array([[6, 9, 0, 1, 6],
       [5, 1, 2, 5, 4],
       [8, 5, 5, 8, 4],
       [0, 9, 2, 7, 5],
       [0, 4, 9, 9, 7]])
w = array([0.2, 0.2, 0.2, 0.2, 0.2])
labels = array([0,1,1,0,0])
```

1. sort the features by value, axis=1, each row will have labels sorted by feature value
```python
f = array([[0, 1, 6, 6, 9],
       [1, 2, 4, 5, 5],
       [4, 5, 5, 8, 8],
       [0, 2, 5, 7, 9],
       [0, 4, 7, 9, 9]])
labels= array([[1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1]
])
```
2. accumulate the weights of misclassifications on the axis=1;
```python
#! Not Sure if this is correct, but, yOu GeT tHe iDeA
# left is negative, right is positive
w1 = array([[0.2, 0.2, 0.2, 0.2, 0.4],
       [0.2, 0.4, 0.4, 0.4, 0.4],
       [0, 0.2, 0.4, 0.4, 0.4],
       [0, 0.2, 0.4, 0.4, 0.4],
       [0, 0.2, 0.2, 0.2, 0.4]])
# left is positive, right is negative - accumulative from the right:
w2 = array([[0.6, 0.6, 0.4, 0.2, 0],
       [0.6, 0.6, 0.6, 0.4, 0.2],
       [0.6, 0.4, 0.4, 0.4, 0.2],
       [0.6, 0.4, 0.4, 0.4, 0.2],
       [0.6, 0.4, 0.4, 0.2, 0]])
```

3. find the index where the sum of `w1` and `w2` is minimum, axis=1
```python
w_sum = w1 + w2
w_sum = array([[0.8, 0.8, 0.6, 0.4, 0.4],
       [0.8, 1, 1, 0.8, 0.6],
       [0.6, 0.6, 0.8, 0.8, 0.6],
       [0.6, 0.6, 0.8, 0.8, 0.6],
       [0.6, 0.6, 0.6, 0.4, 0.4]])
best_f_index = array([3, 4, 0, 0, 0])
```
4. those indecies are the thresholds, repeat for reversed polarity, choose best threshold, polarity and error



</details>

# feat_desc.ipynb  
This is just a test file for the features_description.py, it compares its output with other implementations of the same features