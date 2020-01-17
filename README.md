# Replay Anti-spoofing Countermeasure based on Data Augmentation with Post Selection
This repo is for storing the resources used in the submission about data augmentation with post selection for replay anti-spoofing.

Contents:
1. Features: contains the MATLAB toolbox used for extracting the CQCC/CQT features, also a matlab function used for truncatitng and   normalizing CQT features. 
2. Figures: contains the figs used in the manuscript.
3. Models: contains the training.py for setting and training the GAN networks. The log of the training can be saved as .mat file by log_check.py
4. Post_selection: contains two post filter, one is CNN and another is SVM based. 
5. SD_systems: contains the deep learning based detection system. The GMM baseline system is released by the challenge organizers. 

Detailed user's manual will be complete in the near future. 
For any questions, please open issues or contact Yuanjun Zhao (yuanjun.zhao@research.uwa.edu.au)
