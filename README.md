# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Capstone Project: Pet Behaviour Predictor

### Background

Why Study Dogs Behaviour? First and most important step to building a strong and healthy relationship with your dog is to understand how it thinks, feels, and learns. Understanding your dog and gaining insights into how its mind works can truly make you a better pet parent and companion to it.

A pug may look guilty after tearing up your toilet paper but in actual fact, dogs do not have the capability to feel guilt!

This developmental sequence is the key to understanding the emotions of dogs. Dogs go through emotional developmental stages much more quickly than humans do and have all of the emotional range that they will ever achieve by the time they are four to six months of age.

The assortment of emotions available to the dog will not exceed that which is available to a human who is two to two-and-a-half years old. This means that a dog will have all of the basic emotions: joy, fear, anger, disgust but the dog does not experience the more complex emotions like guilt, pride, and shame.

Reference link - https://moderndogmagazine.com/articles/which-emotions-do-dogs-actually-experience/32883

### Problem Statement

“How can we help new dog owners understand their dog’s overall emotions to foster healthy development of dogs?"

---

### Data Used

[`Dog images from Kaggle`](https://www.kaggle.com/code/sarthakkapaliya/dogemotionrecognition/input): Data is downloaded from Kaggle

---

### Data Dictionary

|Feature|Type|Description|
|---|---|---|
|**train_data**|*jpg*|4 classes of dog emotions:<br>Angry - 2101 images<br>Happy - 2101 images<br>Relaxed - 2100 images<br>Sad - 2100 images|
|**validation_data**|*jpg*|4 classes of dog emotions:<br>Angry - 211 images<br>Happy - 340 images<br>Relaxed - 202 images<br>Sad - 252 images|

---

### Notebook description

* [`01_Project Details`](/code/01%20Project%20Details.ipynb): Background of Project and Problem Statement
* [`02 EDA and Modelling`](/code/02%20EDA%20&%20Models.ipynb): Exploratory data analysis and Modelling of data

---

### Conclusion

- EfficientNet model yields train and cross validation accuracy of 0.99 and 0.85 respectively.
- With this model, we developed an user-friendly predictive tool to help new dog owners identify easily if their pet is sad or relaxed as these two emotions are often hard to differentiate.

---

### How to Foster Healthy Development of your Dog

#### 1 - Increase Social Interactions
- Set up doggy playdates with other canine friends 
#### 2 - Provide a safe space and environment
- A comfortable environment with cozy bed and soft toys keeps the dog feel safe
#### 3 - Incorporate Play time
- Exercises & playtime such as play fetch, brisk walk and giving them toys are positive reinforcement to help reinforce good behaviour
#### 4 - Dog Training
- If your dog is showing signs of agression frequently, consider consulting a professional dog trainer for advice
