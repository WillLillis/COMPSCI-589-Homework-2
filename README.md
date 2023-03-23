My code can be run via the command ``python3 src.py``. The program takes no additional command line arguments.

The code's dependencies are the ``utils.py`` file, the ``log2`` function from the ``math`` module, ``numpy`` (for its extended precision floating point types), and ``randrange`` function from the ``random`` module.

Each run of the code evaluates both the naive and "log trick" implementation of the naive bayes algorithm.

Other aspects of the experiments can be altered via the variables at the top of ``main()``.

- Laplace smoothing can be enabled/disabled by setting the ``g_laplace_smooth`` variable to ``True`` or ``False``.
- The value of α used by said Laplace Smoothing can be changed by setting the value of ``g_alpha``.
- The percentages of the positive and negative training and test sets can be changed by setting the values of ``train_pos_perc``, ``train_neg_perc``, ``test_pos_perc``, and ``test_neg_perc`` accordingly.