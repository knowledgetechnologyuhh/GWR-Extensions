# neunets
This folder contains 3 types of extensions of the Grow-When-Required (GWR) self-organizing neural network by Marsland et al. (2002)

Use **demo** files for off-the-shelf functionalities for self-organizing networks such as: create, train, test, save, import, and plot.

----------------------------------------------------------
**Associative GWR** (AGWR; Parisi et al., 2015) - Standard GWR extended with associative labelling for learning label histograms for each neuron.

----------------------------------------------------------
**GammaGWR** (Parisi et al., 2017) - GWR neurons are equipped with a number of context descriptors for temporal processing.

----------------------------------------------------------
**GammaGWR+** (Parisi et al., 2018) - GammaGWR for incremental learning with temporal synapses and an additional option for regularized neurogenesis. GammaGWR+ implements intrinsic memory replay via recurrent neural activity trajectories.
(check incremental_demo.py)

----------------------------------------------------------
For additional details, please check the reference papers:

[Marsland et al., 2002] Marsland, S., Shapiro, J., and Nehmzow, U. (2002). A self-organising network that grows when required. Neural Networks, 15(8-9):1041-1058.

[Parisi et al., 2015] Parisi, G.I., Weber, C., Wermter, S. (2015) Self-Organizing Neural Integration of Pose-Motion Features for Human Action Recognition. Frontiers in Neurorobotics, 9(3).

[Parisi et al., 2017] Parisi, G.I., Tani, J., Weber, C., Wermter, S. (2017) Lifelong Learning of Human Actions with Deep Neural Network Self-Organization. Neural Networks, 96:137-149.

[Parisi et al., 2018] Parisi, G.I., Tani, J., Weber, C., Wermter, S. (2018) Lifelong Learning of Spatiotemporal Representations with Dual-Memory Recurrent Self-Organization. arXiv:1805.10966.
