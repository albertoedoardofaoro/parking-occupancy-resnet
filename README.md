# Parking Lot Occupancy Estimation with Deep Learning
 
Deep learning project in PyTorch that estimates parking lot occupancy from camera images, developed with [@elisaquadrini](https://github.com/elisaquadrini).
 
A shared **ResNet-18** backbone supports two tasks:
 
- **Regression:** predict the number of occupied parking spaces in an image.
- **Binary classification:** determine whether the parking lot is mostly occupied or mostly empty.
## Dataset
 
The project uses the **PKLot** dataset: camera images of parking areas at the Universidade Federal do Paraná (Brazil), taken under different weather conditions. Each image is annotated in COCO format, with bounding boxes for individual parking spots.
 
From these annotations, two targets were built:
 
- **Regression target:** the number of occupied spaces in the image.
- **Classification target:** a binary label based on the occupancy ratio `ratio_occ = n_occ / n_total`, with a 50% threshold.
## Approach
 
### 1. Data pipeline
- Custom dataset classes return `(image, target)` pairs for regression and `(image, label)` pairs for classification.
- Mini-batches are built with a data loader with configurable batch size and shuffling.
### 2. Exploratory analysis and preprocessing
- Checked the distribution of the binary labels for class imbalance.
- Preprocessing: resizing, tensor conversion, normalization, and color jittering to improve robustness to lighting and weather conditions.
### 3. Architecture
- **Backbone:** ResNet-18, producing a 512-dimensional feature vector.
- **Regression head:** a single linear layer with a scalar output.
- **Classification head:** two fully connected layers with ReLU activations and dropout, producing a scalar logit.
### 4. Regression training
- Mean squared error loss, Adam optimizer, Kaiming (He) initialization.
- Learning rate selected with a learning rate finder, then warm-up followed by cosine annealing.
- Early stopping on validation loss; the model with the lowest validation loss is restored.
### 5. Regression evaluation
- MSE, RMSE and MAE.
- A relative error measure (ROE).
- MAE across occupancy ranges, to check robustness at low and high occupancy.
### 6. Classification training (transfer from regression)
- Reuses the ResNet-18 backbone trained for regression, with its parameters frozen.
- Replaces the regression head with the binary classification head.
- Binary cross-entropy loss, Adam optimizer with cosine annealing, early stopping on validation accuracy.
### 7. Classification evaluation
On a held-out test set:
- Confusion matrix.
- Accuracy, precision, recall and F1-score.
- ROC curve and AUC.
### 8. Model interpretation
Grad-CAM and Ablation-CAM highlight the image regions that contribute most to the predictions, for both tasks.
 
## Repository contents
 
- Scripts for dataset construction, training and evaluation.
- Trained models and checkpoints.
- A presentation describing the full workflow.
## Tech stack
 
Python, PyTorch, torchvision.
