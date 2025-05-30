# Enhancing Breast Cancer PET/CT Resolution Using GANs and Diffusion Models

This project enhances ultra-low-dose PET images in breast cancer(breast cancer dataset from TCIA) cases by fusing CT guidance via a custom U-Net-inspired architecture named ModelILGenerator. The model was developed as part of a generative AI and deep learning capstone course project. We benchmarked our architecture against diffusion-based models and classical GAN setups, achieving superior SSIM and PSNR performance while ensuring anatomical fidelity.

# Main Libraries Used
- PyTorch
- NumPy
- pydicom
- matplotlib
- scikit-image

# Summary
- Designed and implemented ModelILGenerator, a task-specific PET/CT fusion model.
- Adapted low-dose simulation logic for realism.
- Conducted visual and metric-based comparisons with existing diffusion pipelines.
