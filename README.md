# machine learning for art attribution
## Code associated with the work "Discerning the painter’s hand:  machine learning on surface topography".
## Introduction
Attribution of painted works is a critical problem in art history. In painted works, surface topography adds important unintentional stylistic components such as the deposition and drying of the paint, in addition to the intentional brushwork of the artist. Current art historical scholarship includes a growing emphasis on workshop practice used in the creation of paintings by renowned artists. Many famous artists, including El Greco, Rembrandt, and Peter Paul Rubens, utilized workshops to meet market demands for their art. In this study, we hypothesize that significant unintentional stylistic information exists in the 3D surface structure embedded by the painter during the painting process, and that this can be captured through optical profilometry. In addition, by investigating patches at much smaller length scales than any recognizable feature of the painting, differences in the artist’s intrinsic and unintentional style will be revealed. We examine these hypotheses using a controlled study of bespoke paintings by several artists that mimic certain workshop practices. 

A series of twelve paintings by four artists, and their associated topographical profiles, is subject to analysis to attribute the works and to ascertain the important properties involved in that attribution. To analyze these profiles, we use convolutional neural networks (CNNs) to provide a robust and quantitative method to distinguish among random small areas of the triplicate paintings by multiple artists. Bi-directional multivariate empirical mode decomposition (EMD) is applied to the height data collected by profilometry to separate the data into complementary intrinsic mode functions (IMFs) of varying spatial frequency. Finally, we compare ML on photographic and topographic data.  

