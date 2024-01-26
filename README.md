# DiffRS-net ：Multi-Task Learning Framework for Classifying Breast Cancer Subtypes on Multi-Omics data
# Abstract
The precise classification of breast cancer subtypes is crucial for clinical diagnosis and treatment, yet early symptoms are often subtle. Utilizing multi-omics data from high-throughput sequenc-ing can improve the classification accuracy. However, most research primarily focus on the as-sociation between individual omics data and breast cancer, neglecting the interactions between different omics. This may fail to provide a comprehensive understanding of the biological pro-cesses of breast cancer. Here we propose a novel framework called DiffRS-net for classifying breast cancer subtypes by identifying the association among different omics. DiffRS-net per-forms differential analysis on each omics data to identify differentially expressed genes (DE-genes) and adopts a robustness-aware sparse typical correlation analysis to detect multi-way association among DE genes. These DE-genes with high levels of correlation are then used to train a multi-task neural network, thereby enhancing the prediction accuracy of breast cancer subtypes. Experimental results show that by mining the associations between multi-omics data, DiffRS-net achieves more accurate classification of breast cancer subtypes than the existing methods.

## In the robustness-aware adaptive SMCCA model：
  Run "Running.m" to get the result of the Multi-way association analysis of the mult-omics data.
  <br>"rAdaSMCCA.m" is the function of robustness-aware adaptive SMCCA model.
  <br>"normalize.m" is the function to normalize the input data.
  <br>"updataD.m" is the function to iterate the weights.
## In the multi-task learning model:
  Run "binaryC.py" to get the performance of binary classification of DiffRS-net.
  <br>Run "multiC.py" to get the performance of multi-classification of DiffRS-net.
  <br> Contact: If you have any questions or suggestions with the code, please let us know. Contact Yiran Huang at hyr@gxu.edu.cn
## about:
title = {Multi-Task Learning Framework for Classifying Breast Cancer Subtypes on Multi-Omics data)
<br>  
