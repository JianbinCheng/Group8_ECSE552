# Deep learning-Based Plant Disease Detection Using Lightweight Vision Transformers (LeViT)

This repository contains the code for our final project on automated plant disease detection using a dual-stream LeViT-192 architecture. The project focuses on detecting diseases on tomato leaves by leveraging two complementary image preprocessing streams—color jittering and K-means segmentation—and fusing their extracted features via a lightweight Vision Transformer. In addition, extensive ablation studies are performed to evaluate the contribution of each data stream and of key LeViT components.

## Repository Contents

The repository includes the following Jupyter Notebook files:

1. **Model.ipynb**  
   - **Description:** Contains the implementation and training code for the baseline dual-stream LeViT model.  
   - **Key Points:**  
     - Implements the dual-stream architecture where one stream processes color jittered images (original) and the other processes K-means segmented images.  
     - Uses a FusionMLP to combine features from both streams.  
     - Includes data preprocessing, training, validation, and testing scripts.

2. **Tesing_alternatives.ipynb**  
   - **Description:** Contains experiments and comparisons with alternative architectures and settings.  
   - **Key Points:**  
     - Provides additional experiments and diagnostic plots.
     - Tests different hyperparameters and training settings.

3. **K_mean_Preprocessing.ipynb**  
   - **Description:** Implements the K-means segmentation preprocessing pipeline for plant leaf images.  
   - **Key Points:**  
     - Applies K-means clustering (with a specified K value) to segment images and emphasize disease-related regions.
     - Generates segmented images that are fed to the model in the dual-stream framework.

4. **Ablation_LeViT192_Components.ipynb**  
   - **Description:** Contains ablation experiments on the internal components of the LeViT-192 model.  
   - **Key Points:**  
     - Each transformer block (termed as Shrink Attention Block in our report) includes three components:  
       - **LevitBlock (Shrink Attention Block)**
       - **Attention (MHSA Block)**
       - **LevitMLP (MLP Block)**
     - Runs experiments masking one component type at a time (by replacing its forward function with an identity mapping) and compares the resulting performance.

5. **Ablation_DualStream_maskingColor_keepK.ipynb**  
   - **Description:** Implements an ablation experiment on the dual-stream structure in which the color-jitter (original) stream is masked.  
   - **Key Points:**  
     - During training, only the k-means segmented images are used.
     - During validation and testing, however, the original images (color jittered) are used.
     - This experiment assesses the importance of the color-jitter stream in the dual-stream architecture.

6. **Ablation_DualStream_maskingK_keepColor.ipynb**  
   - **Description:** Implements the complementary ablation experiment where the k-means segmented stream is masked.  
   - **Key Points:**  
     - Training and evaluation in this notebook use only the original (color-jittered) images.
     - This enables us to gauge the contribution of the k-means segmentation.

## How to Run

1. **Setup Environment:**  
   Ensure that you have Python 3 installed with the following packages:
   - PyTorch
   - timm
   - tqdm
   - numpy
   - Jupyter Notebook (or an environment like Google Colab)

2. **Data Preparation:**  
   Make sure your dataset is organized as required (the PlantVillage tomato leaf dataset).  
   The notebooks assume that your data loaders (e.g., `train_loader`, `val_loader`, and `test_loader`) are properly defined within the notebook.

3. **Running Notebooks:**  
   - Open the desired notebook in Jupyter Notebook or Google Colab.
   - Follow the instructions within the notebook to train and evaluate the model.  
   - For ablation experiments, each notebook is self-contained and explains which stream or component is masked.

## Code Overview

### Dual-Stream Model (Model.ipynb)
- **Architecture:**  
  Two LeViT-192 feature extractors feed their outputs into a FusionMLP.
- **Preprocessing:**  
  One stream uses color jittering, and the other uses K-means segmentation.
- **Training/Testing:**  
  Standard training/validation loops with early stopping are implemented.

### Ablation Studies
- **Ablation of LeViT-192 Components (Ablation_LeViT192_Components.ipynb):**  
  - Masks entire blocks, attention modules, or MLP modules by replacing their forward functions with an identity mapping.
  - Compares validation and test performance across these configurations.
- **Ablation of Dual-Stream Structure:**  
  - **Ablation_DualStream_maskingColor_keepK.ipynb:**  
    - Trains the model using only k-means segmented images (masking the color jitter stream).
    - For validation/testing, the original images are used.
  - **Ablation_DualStream_maskingK_keepColor.ipynb:**  
    - Trains the model using only the original images (masking the k-means stream).
    - This allows us to understand the impact of each input stream on overall performance.

## Results Summary

Our final report discusses the following key findings:
- The dual-stream LeViT model achieves high accuracy while balancing computational efficiency.
- Ablation studies of the LeViT components indicate that:
  - Masking the entire block (Shrink Attention Block) sometimes improves performance due to redundancy.
  - Masking just the attention modules leads to a slight drop.
  - Masking the MLP modules causes the most pronounced decrease, highlighting their role in feature refinement.
- Ablation of the input streams shows that both the color-jitter and k-means streams contribute unique strengths to the model.

## Contact

For any questions regarding the code or experiments, please contact:
- Tingrui Zhang: [tingrui.zhang@mail.mcgill.ca](mailto:tingrui.zhang@mail.mcgill.ca)
- Yushu Zhao: [yushu.zhao@mail.mcgill.ca](mailto:yushu.zhao@mail.mcgill.ca)
- Jianbin Cheng: [jianbin.cheng@mail.mcgill.ca](mailto:jianbin.cheng@mail.mcgill.ca)
