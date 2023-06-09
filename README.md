# Functional PDE Regression
## AMATH 563 Spring 2023

Included is all python files and notebooks used to produce results of final report. 

We used scikit-learn [7] for all kernel matrix computations. To implement kernel differentiation, we used Jax [2] to estimate our derivatives. Numpy [4] was additionally used for all matrix-based compuations. When comparing the kernel approach to the SINDy approach, we used the PySINDy package [3, 6] for all SINDy-related computations and model training. Finally, to obtain ground-truth measurements for the Darcy flow system, we used FEniCS from FEM on Colab [1] to numerically solve the Darcy flow PDE. All generated plots were created using matplotlib [5].

## References
[1] F. Ballarin. FEM on colab. https://fem-on-colab.github.io, 2022.

[2] J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas,
S. Wanderman-Milne, and Q. Zhang. JAX: composable transformations of Python+NumPy programs. http://github.com/google/jax, 2018.

[3] B. de Silva, K. Champion, M. Quade, J.-C. Loiseau, J. Kutz, and S. Brunton. Pysindy: A python package for
the sparse identification of nonlinear dynamical systems from data. Journal of Open Source Software, 5(49):2104,
2020.

[4] C. R. Harris, K. J. Millman, S. J. van der Walt, R. Gommers, P. Virtanen, D. Cournapeau, E. Wieser, J. Taylor,
S. Berg, N. J. Smith, R. Kern, M. Picus, S. Hoyer, M. H. van Kerkwijk, M. Brett, A. Haldane, J. F. del R ́ıo,
M. Wiebe, P. Peterson, P. G ́erard-Marchant, K. Sheppard, T. Reddy, W. Weckesser, H. Abbasi, C. Gohlke, and
T. E. Oliphant. Array programming with NumPy. Nature, 585(7825):357–362, Sept. 2020.

[5] J. D. Hunter. Matplotlib: A 2d graphics environment. Computing in Science & Engineering, 9(3):90–95, 2007.

[6] A. A. Kaptanoglu, B. M. de Silva, U. Fasel, K. Kaheman, A. J. Goldschmidt, J. Callaham, C. B. Delahunt, Z. G.
Nicolaou, K. Champion, J.-C. Loiseau, J. N. Kutz, and S. L. Brunton. Pysindy: A comprehensive python package
for robust sparse system identification. Journal of Open Source Software, 7(69):3994, 2022.

[7] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss,
V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn:
Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830, 2011.
