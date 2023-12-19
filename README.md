Code for the following Paper:

Li J, Meng Y, Tao C, et al. ConvFormerSR: Fusing Transformers and Convolutional Neural Networks for Cross-sensor Remote Sensing Imagery Super-resolution[J]. IEEE Transactions on Geoscience and Remote Sensing, 2023.

[[Paper](https://ieeexplore.ieee.org/document/10345595)]

# Abstract

Super-resolution (SR) techniques based on deep learning have a pivotal role in improving the spatial resolution of images. However, remote sensing images exhibit ground objects characterized by diverse types, intricate structures, substantial size discrepancies, and noise. Simultaneously, variations in imaging mechanisms, imaging time, and atmospheric conditions among different sensors result in disparities in image quality and surface radiation. These factors collectively pose challenges for existing SR models to fulfill the demands of the domain. To address these challenges, we propose a novel cross-sensor SR framework (ConvFormerSR) that integrates transformers and convolutional neural networks (CNNs), catering to the heterogeneous and complex ground features in remote sensing images. Our model leverages an enhanced transformer structure to capture long-range dependencies and high-order spatial interactions, while CNNs facilitate local detail extraction and enhance model robustness. Furthermore, as a bridge between the two branches, a feature fusion module (FFM) is devised to efficiently fuse global and local information at various levels. Additionally, we introduce a spectral loss based on the remote sensing ratio index to mitigate domain shift induced by cross-sensors. The proposed method is validated on two datasets and compared against existing state-of-the-art SR models. The results show that our proposed method can effectively improve the spatial resolution of Landsat-8 images, and the model performance is significantly better than other methods. Furthermore, the SR results exhibit satisfactory spectral consistency with high-resolution (HR) images.

# The overall architecture

ConvFormerSR first obtains the shallow feature of LR through a convolutional layer, and then obtains the deep features through the parallel transformer branch and CNN branch, respectively, and fuses the features of the two branches through the FFM module. Subsequently, a global residual connection is used to fuse low-level features and deep features, and finally the pixel-shuffle is used for upsampling to obtain SR results.

![fig1](/fig/fig1.png)

# Result

Qualitative comparison of our model with other works on the HLSSR-GJ dataset.

![fig2](/fig/fig2.png)

# Training and Testing

Please use [BasicSR](https://github.com/XPixelGroup/BasicSR), it is an open-source image and video restoration toolbox based on PyTorch.

# Citation 

If you find this work useful, please consider citing:

```bibtex
@ARTICLE{10345595,
  author={Li, Junjie and Meng, Yizhuo and Tao, Chongxin and Zhang, Zhen and Yang, Xining and Wang, Zhe and Wang, Xi and Li, Linyi and Zhang, Wen},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={ConvFormerSR: Fusing Transformers and Convolutional Neural Networks for Cross-Sensor Remote Sensing Imagery Super-Resolution}, 
  year={2024},
  volume={62},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2023.3340043}}
```

# Reference

Some of the codes in this repo are borrowed from:  
- [SwinIR](https://github.com/JingyunLiang/SwinIR)  
- [BasicSR](https://github.com/XPixelGroup/BasicSR)  
- [DANet](https://github.com/junfu1115/DANet)  
- [CBAM](https://github.com/luuuyi/CBAM.PyTorch)  




