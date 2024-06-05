# Statistical analysis of features from usg images

## Aim
To determine whether it is possible to classify ultrasound images of the skin in relation to disease (Actinic Keratosis, AK) severity based on morphological and textural features


## Stages
1. Search for features for which there are differences between groups - stages of disease (AK1, AK2, AK3) for features that are easy to interprete by doctors (Krusgall-Wallis for detecting if there are any differences with eta squared calculated and Dunn's post hoc to determinate between which groups)
3. Sorting of features according to their statistical significance (MRMR) and oversampling smallest groups (SMOTE)
4. Selection of classifiers and iterative search of the cut-off point for the ordered features and search for parameters
5. Heatmaps with classification results

