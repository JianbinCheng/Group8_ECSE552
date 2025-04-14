# Deep learning-Based Plant Disease Detection Using Lightweight Vision Transformers (LeViT)

This repository contains the code for the final project on automated plant disease detection using a dual-stream LeViT-192 architecture. The project focuses on detecting diseases on tomato leaves by leveraging two complementary image preprocessing streams—color jittering and K-means segmentation—and fusing their extracted features via a lightweight Vision Transformer. In addition, extensive ablation studies are performed to evaluate the contribution of each data stream and of key LeViT components.

## Repository Contents

The repository includes the following Jupyter Notebook files:

1. **Model.ipynb**  
   - **Description:** Contains the implementation and training code for the dual-stream LeViT model.  
   

2. **Tesing_alternatives.ipynb**  
   - **Description:** Contains experiments and comparisons with alternative architectures and settings.  
   

3. **K_mean_Preprocessing.ipynb**  
   - **Description:** Implements the K-means segmentation preprocessing pipeline for plant leaf images.  
   
4. **Ablation_LeViT192_Components.ipynb**  
   - **Description:** Contains ablation experiments on the internal components of the LeViT-192 model.  
   

5. **Ablation_DualStream_maskingColor_keepK.ipynb**  
   - **Description:** Implements an ablation experiment on the dual-stream structure in which the color-jitter (original) stream is masked.  
   

6. **Ablation_DualStream_maskingK_keepColor.ipynb**  
   - **Description:** Implements the complementary ablation experiment where the k-means segmented stream is masked.  
   

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
     
4. **Trained Model:**
   You can download at:https://drive.google.com/drive/folders/1GGNjH5sZPmiWgQqWv6OXqmabO6xiGzMl?usp=sharing.

## Contact

For any questions regarding the code or experiments, please contact:
- Tingrui Zhang: [tingrui.zhang@mail.mcgill.ca](mailto:tingrui.zhang@mail.mcgill.ca)
- Yushu Zhao: [yushu.zhao@mail.mcgill.ca](mailto:yushu.zhao@mail.mcgill.ca)
- Jianbin Cheng: [jianbin.cheng@mail.mcgill.ca](mailto:jianbin.cheng@mail.mcgill.ca)
