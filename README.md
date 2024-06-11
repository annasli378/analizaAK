# Statistical analysis of features from medical imaging of Actinic Keratosis

## Aim
To determine whether it is possible to classify ultrasound images of the skin in relation to disease (Actinic Keratosis, AK) severity based on morphological and textural features

## Data
The dataset consisted of 163 pairs of AK lesion images (dermatoscopy and 20MHz HFUS) from 53 patients. The evaluation - classification into groups 1, 2 or 3 - was performed by 2 experts.

## Stages
### Feature extraction:
Feature extraction: from ultrasound images and from dermatoscopic images [link will appear later].

Features were normalized using z-score normalization.

### Exploring differences between groups Notebook analysis_features.ipynb and script Boxplot_interpretable_AK_features.R
Search for features for which there are differences between groups - stages of disease (AK1, AK2, AK3).  Particular attention has been paid to features where a doctor's interpretation is possible. The Krusgall-Wallis test was used to detect differences between groups. In addition, scale effects were determined (eta squared) and Dunn's post hoc test was used to determinate between which groups the differences were significant.

### Preparation for classification
The features were then sorted using the Maximum-Relevance Minimum-Redundancy algorithm. Next, SMOTE technique was applied for oversampling the smallest groups.

### Classification gridsearch.py
Leave-one-out according to patients was performed - one patient in each iteration is added to the test set, training on the all remaining. An SVM classifier was chosen and then iteratively searched for the optimal cut-off point for the ranked data, as well as the optimal values of the control parameters (C and gamma). In addition, Cohen's Kappa values were determined.

In addition, various features were tested: 
 -  HFUS (handcrafted)
 -  Dermatoscopic (handcrafted)
 -  Dermatoscopic (extracted from Neural Network)
 -  Dermatoscopic (handcrafted and extracted from Neural Network)
 -  HFUS & Dermatoscopic handcrafted
 -  HFUS & Dermatoscopic  from Neural Network
 -  HFUS & Dermatoscopic handcrafted & from Neural Network

## Results
Currently, the best results were obtained for the features obtained from the network alone (81 accuracy, 74 kappa, number of features equal 42). However, further research is ongoing and will appear later 

### Main limitations of the method:
- too few samples, especially from group 3,
- no check on the influence of the test site on the results,
- not checking how the characteristics depend on the patient himself

# Bibliography

[MRMR] https://www.mathworks.com/help/stats/fscmrmr.html

[SMOTE] https://arxiv.org/pdf/1106.1813

[SVM] https://scikit-learn.org/stable/modules/svm.html
