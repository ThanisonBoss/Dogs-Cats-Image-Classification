import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# View an image
def viewRandomImage(targetDir, targetClass, num=5):
    """Explor randomly image
    
    Args:
        targetDir (str): Directory path of image
        targetClass (list): list str of class
        num (int, optional): amount of image to exploration 
    
    Return:
        image: Random image 
    
    """

    plt.figure(figsize=(10,10))
    for i in range(num):
        randomClass = random.sample(targetClass,1)[0]
        targetFloder = targetDir + randomClass + '/'
        randomImage = random.sample(os.listdir(targetFloder),num)
        img = mpimg.imread(targetFloder+randomImage[i])
        plt.subplot(1,num,i+1),plt.imshow(img)
        plt.axis("off")
        plt.title(randomClass)

def lossPlot(history):
    """Plot loss and accuracy between training
    Args: 
        history (dict) : dictionary fron callbach history between training model

    Return:
        visulization line plot of loss and accuracy
    """
    trainLoss = history.history['loss']
    testLoss = history.history['val_loss']
    trainAccuracy = history.history['accuracy']
    testAccuracy = history.history['val_accuracy']

    # Plot Loss
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    plt.plot(trainLoss, label="train_loss")
    plt.plot(testLoss, label="test_loss")
    plt.title("loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1,2,2)
    plt.plot(trainAccuracy, label="train_accuracy")
    plt.plot(testAccuracy, label="test_accuracy")
    plt.title("accuracy")
    plt.legend()