# Sentiment Analysis on Tweets
This is a project under "Data Science Honors" degree at my University: Pune Instititute of Computer Technology.

## Requirements

The general requirements are as follows.  
* `numpy`
* `scikit-learn`
* `scipy`
* `nltk`

The library requirements specific to some methods are:
* `keras` with `TensorFlow` backend for Logistic Regression, MLP, RNN (LSTM), and CNN.

### Baseline
3. Run `baseline.py`. With `TRAIN = True` it will show the accuracy results on training dataset.

### Naive Bayes
4. Run `naivebayes.py`. With `TRAIN = True` it will show the accuracy results on 10% validation dataset.

### Maximum Entropy
5. Run `logistic.py` to run logistic regression model OR run `maxent-nltk.py <>` to run MaxEnt model of NLTK. With `TRAIN = True` it will show the accuracy results on 10% validation dataset.

### Decision Tree
6. Run `decisiontree.py`. With `TRAIN = True` it will show the accuracy results on 10% validation dataset.

### Random Forest
7. Run `randomforest.py`. With `TRAIN = True` it will show the accuracy results on 10% validation dataset.

### XGBoost
8. Run `xgboost.py`. With `TRAIN = True` it will show the accuracy results on 10% validation dataset.

### SVM
9. Run `svm.py`. With `TRAIN = True` it will show the accuracy results on 10% validation dataset.

### Multi-Layer Perceptron
10. Run `neuralnet.py`. Will validate using 10% data and save the best model to `best_mlp_model.h5`.

### Reccurent Neural Networks
11. Run `lstm.py`. Will validate using 10% data and save models for each epock in `./models/`. 

### Convolutional Neural Networks
12. Run `cnn.py`. This will run the 4 conv layers neural network model as described in the report. To run other versions of CNN, just comment or remove the lines where Conv layers are added. Will validate using 10% data and save models for each epoch in `./models/`. 


