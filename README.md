# Enhancing Breast Cancer PET/CT Resolution Using GANs and Diffusion Models

This project enhances ultra-low-dose PET images in breast cancer(breast cancer dataset from TCIA) cases by fusing CT guidance via a custom U-Net-inspired architecture named ModelILGenerator. The model was developed as part of a generative AI and deep learning capstone course project. We benchmarked our architecture against diffusion-based models and classical GAN setups, achieving superior SSIM and PSNR performance while ensuring anatomical fidelity.

# Main Libraries Used
- PyTorch
- NumPy
- pydicom
- matplotlib
- scikit-image

#File Structure
PET_SR_PROJECT/
├── genAI/
│   ├── data/
│   │   ├── raw_dicom/
│   │   ├── high_dose/
│   │   └── low_dose/
│   ├── datasets/
│   │   └── pet_dataset.py
│   ├── models/
│   │   ├── model_il.py             ← Custom ModelILGenerator architecture
│   │   ├── discriminator.py        ← PatchGAN
│   │   └── simple_sr.py            ← Alternative baseline model
│   ├── train_fusion_gan.py         ← Main training loop
│   ├── simulate_low_dose_fusion.py← Simulation code (Poisson noise)
│   ├── convert_pet_ct_pair.py      ← DICOM to NumPy preprocessor
│   ├── evaluate.py                 ← Evaluation metrics + visualization
│   └── plot_losses.py              ← Training curve visualization
├── generator_gan.pth               ← Pretrained model weights
├── final_training_loss.png         ← Loss graph
└── README.md                       ← 

#Summary
- Designed and implemented ModelILGenerator, a task-specific PET/CT fusion model.
- Adapted low-dose simulation logic for realism.
- Conducted visual and metric-based comparisons with existing diffusion pipelines.
