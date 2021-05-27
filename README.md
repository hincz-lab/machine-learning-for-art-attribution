# machine learning for art attribution
## Code associated with the work "Discerning the painter’s hand:  machine learning on surface topography".
## Introduction
Attribution of painted works is a critical problem in art history. In painted works, surface topography adds important unintentional stylistic components such as the deposition and drying of the paint, in addition to the intentional brushwork of the artist. Current art historical scholarship includes a growing emphasis on workshop practice used in the creation of paintings by renowned artists. Many famous artists, including El Greco, Rembrandt, and Peter Paul Rubens, utilized workshops to meet market demands for their art. In this study, we hypothesize that significant unintentional stylistic information exists in the 3D surface structure embedded by the painter during the painting process, and that this can be captured through optical profilometry. In addition, by investigating patches at much smaller length scales than any recognizable feature of the painting, differences in the artist’s intrinsic and unintentional style will be revealed. We examine these hypotheses using a controlled study of bespoke paintings by several artists that mimic certain workshop practices. 

In our experiment, a series of twelve paintings by four artists, and their associated topographical profiles, is subject to analysis to attribute the works and to ascertain the important properties involved in that attribution. To analyze these profiles, we use convolutional neural networks (CNNs) to provide a robust and quantitative method to distinguish among random small areas of the triplicate paintings by multiple artists. Bi-directional multivariate empirical mode decomposition (EMD) is applied to the height data collected by profilometry to separate the data into complementary intrinsic mode functions (IMFs) of varying spatial frequency. Finally, we compare ML on photographic and topographic data.  
![sample_painting](https://user-images.githubusercontent.com/24704249/119850155-4504f480-bedb-11eb-84d7-a65948fc4d2c.png)
Each artist (artist 1-4) created three painting, one of which was reserved for testing the training CNN algorithm. These test paintings are shown for all four artist as first row: high-resolution photographs (all paintings used in the study can be found in SI in our paper), and second row: height data, shaded in grayscale from low (darker) to high (lighter). 
## Exploring the effect of patch size on attribution accuracy
In preparation for the experiments, the height information is digitally split into small patches with a range of patch sizes from 10 pixels (0.5 mm) to 1200 pixels (6 cm). We train those patches using the pretrained VGG-16 network. The following figure shows sample patches of height data from each artist, each patch is 200x200 pixels (10x10 mm).
<p align="center">
<![sample_patches](https://user-images.githubusercontent.com/24704249/119850265-6534b380-bedb-11eb-9b63-568f48de51a4.png)>

## Using empirical mode decomposition to determine the length-scales of the brushstroke topography 
![imf](https://user-images.githubusercontent.com/24704249/119856447-b6937180-bee0-11eb-8db0-5e3727a46acc.png)

## Comparing topography versus photography when testing on data with novel characteristics
![fore_back](https://user-images.githubusercontent.com/24704249/119856399-ad0a0980-bee0-11eb-913d-76c62936e251.png)
