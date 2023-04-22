- Training
  - X is copied multiple times in classifier.py, so it is better to make it a global variable??
  - Build architecture like the paper to deduce T, layers count
  - Train a hard dataset
  - How to make processing features in batches, each batch from a different file?
  - Implement f value on GPU (chosing 10%)
--------------
- Testing
  - Test using chosen features only
    - Pass exact needed feature for each image instead of all features for each image
      - This needs to be different in `classifier.py` than in `strong_classifier.py`, than in `cascade.py`
  - Detect face in a general image (subwindows, scaling)
  - Detect faces in a video
  - Normalize faces? (μ, σ)
  - Delete `Please Call chooseClassifier() first` warning?

--------------
README.md

--------------
- Refactor code
- cpu instead of gpu
- num_workers in dataloader