# CIFAR10 Image Classification using Transfer Learning

<h1>Description</h1>
<p>This project explores transfer learning for image classification on the CIFAR10 dataset. Transfer learning builds on the knowledge acquired from pre-trained models on large datasets (e.g., ImageNet) and fine-tunes them for a specific task.</p>

<h1>Project Description</h1>

The CIFAR10 dataset contains 60,000 32x32 color images belonging to 10 classes. This project aims to:

* Load and preprocess the CIFAR10 dataset.
* Utilize a pre-trained CNN models
* Train the model on the CIFAR10 dataset.
    * VGG16
    * ResNet34
    * Convext Tiny
* Freeze initial layers of the pre-trained model.
* Fine-tune the final layers for the CIFAR10 dataset.
* Evaluate the model's performance

<h1>Dependencies</h1>

* Python 3.x
* Pytorch 
* Scikit-learn
* torchvision
* Matplotlib
* Numpy
* timm

<h1>Key Concepts</h1>

* **Transfer Learning:** Leverages the knowledge of a pre-trained model to improve performance on a different but related task.
* **Fine-tuning:** Adapting the final layers of a pre-trained model for a new dataset.
* **Dynamic Learning Rate Optimization:** Adjusting the learning rate during training to improve convergence speed and stability.

<h1>Results</h1>

The following table shows the results of the model training and validation on the CIFAR10 dataset.

| Model | Train Accuracy | Validation Accuracy | Test Accuracy |
| --- | --- | --- | --- |
| VGG16 | 96.3% | 92.94% | 92.5% |
| ResNet34 | 97.7% | 94.7% | 94.7% |
| Convext Tiny | 96.5% | 95.7% | 95.7% |
