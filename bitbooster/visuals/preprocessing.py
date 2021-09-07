# import matplotlib.pyplot as plt
# import pandas as pd
# from matplotlib.patches import Rectangle
# from scipy.stats import norm
#
# from bitbooster.common import *
# from bitbooster.preprocessing import discretizers
# from bitbooster.strings import HT1B, JENKS, EQUAL, EQUAL_F
#
#
# def create_plot_with_all_discretizers(frequency_sr=35, n_bins=4, loglog=False, add_normal_overlay=False,
#                                       fuzzy_classes=False, **kwargs):
#     """
#     Visualize the concept of discretization
#
#     Parameters
#     ----------
#     frequency_sr: pd.Series or int
#         Series with frequencies of each index. If int, a theoretical power law with exponent -1 and values
#         1 ... `frequency_sr' is used.
#     n_bins: int
#         Number of bins to use. Use by all algorithms except HT1
#     loglog: Bool
#         Whether to plot a log-log plot. Especially useful for power-law distributions
#     add_normal_overlay: Bool
#         Add a normal distribution to each bin, with mean as the bin centre and std as 1/4 of the bin width
#     fuzzy_classes: Bool
#         Whether to show fuzzy classes (slow progression from one class to another near bin edges), base on the
#         normal_distribution
#
#     Other Parameters
#     ----------------
#     f: float
#         Parameter for the RatioEqualCutDiscretizer
#     """
#     # GENERATE DATA
#     if isinstance(frequency_sr, int):
#         real_values = False
#         n = frequency_sr
#         np.random.seed(n)
#         kwargs['f'] = 0.9
#         p = np.power(np.arange(1, n + 1, dtype=float), -1)
#         p /= sum(p)
#         frequency_sr = pd.Series(index=np.arange(1, n + 1), data=p * 10000).astype(int)
#     else:
#         real_values = True
#
#     # ALGORITHMS
#     disc_algorithms = {
#         # 'None': None,
#         # QCUT: discretizers.QCutDiscretizer(n_bins),
#         EQUAL: discretizers.EqualCutDiscretizer(n_bins),
#         EQUAL_F: discretizers.RatioEqualCutDiscretizer(n_bins, kwargs.get('f', 0.99999)),
#         JENKS: discretizers.JenksDiscretizer(n_bins),
#         # HT1: discretizers.HeadTailsBreakDiscretizer(),
#         HT1B: discretizers.HeadTailsBreakBinsDiscretizer(n_bins),
#     }
#
#     # CREATE A PLOT FOR EACH ALGORITHM
#     for alg_name, alg in disc_algorithms.items():
#         f, ax = plt.subplots()
#
#         if loglog:
#             # Plot on a log-log scale
#             x = np.log2(frequency_sr.index.to_numpy())
#             y = np.log2(frequency_sr.to_numpy())
#
#             pre = rf'$\log_2$('
#             post = ')'
#
#         else:
#             # Plot on a normal scale
#             x = frequency_sr.index.to_numpy()
#             y = frequency_sr.to_numpy()
#
#             pre = post = ''
#
#         if real_values:
#             y_min_label = f'{pre}{min(frequency_sr) / sum(frequency_sr) * 100:.0f}%{post}'
#             y_max_label = f'{pre}{max(frequency_sr) / sum(frequency_sr) * 100:.0f}%{post}'
#             x_min_label = f'{pre}{min(frequency_sr.index)}{post}'
#             x_max_label = f'{pre}{max(frequency_sr.index)}{post}'
#         else:
#             y_min_label = f'{pre}min{post}'
#             y_max_label = f'{pre}max{post}'
#             x_min_label = f'{pre}min{post}'
#             x_max_label = f'{pre}max{post}'
#
#         # Plot data
#         ax.plot(x, y, 'b.')
#
#         # Fix axes
#         ax.set_xlim([min(x), max(x)])
#         ax.set_xticks([min(x), max(x)])
#         ax.set_xticklabels([x_min_label, x_max_label])
#         ax.set_xlabel(f'{pre}value{post}')
#         ax.set_ylim([min(y), max(y)])
#         ax.set_yticks([min(y), max(y)])
#         ax.set_yticklabels([y_min_label, y_max_label])
#         ax.set_ylabel(f'{pre}frequency{post}')
#
#         # Regular data plot
#         if alg is None:
#             continue
#
#         # Train discretization algorithm
#         alg.fit_from_frequencies(frequency_sr)
#
#         # Bin widths
#         widths = alg.ranges[1:] - alg.ranges[:-1]
#
#         # Normal distributions
#         centres = widths / 2 + alg.ranges[:-1]
#         stds = widths / 4
#         pdfs = [norm(loc=centres[i], scale=stds[i]) for i in range(alg.number_of_bins)]
#
#         if fuzzy_classes:
#             # Idea about making bin membership fuzzy/probabilistic. Datapoints near the edge have a probability to be
#             # part of the other bin.
#
#             if loglog:
#                 raise NotImplementedError('Did not implement for loglog AND fuzzy_classes')
#
#             # Create infinitesimal patches
#             n = 1000
#             x_lefts = np.linspace(min(x), max(x), num=n, endpoint=True)
#             labels = alg.transform(x_lefts)
#
#             # Add each patch
#             for i in range(n - 1):
#                 lab = labels[i]
#                 xl = x_lefts[i]
#                 xr = x_lefts[i + 1]
#                 # LEFT BIN
#                 if lab == 0:
#                     frac_left = 0
#                 else:
#                     frac_left = pdfs[lab - 1].pdf(xl)
#
#                 # RIGHT BIN
#                 if lab == alg.number_of_bins - 1:
#                     frac_right = 0
#                 else:
#                     frac_right = pdfs[lab + 1].pdf(xl)
#
#                 # SELF BIN
#                 frac_self = pdfs[lab].pdf(xl)
#                 total_frac_self = frac_self / (frac_self + frac_left + frac_right)
#
#                 # APPLY COLOUR
#                 if lab % 2 != 0:
#                     total_frac_self = 1 - total_frac_self
#
#                 # ADD PATCH FOR DX
#                 ax.add_patch(
#                     Rectangle(xy=(xl, 0), width=xr - xl, height=max(y), alpha=min(1., max(0., 250 / n)),
#                               color=(total_frac_self, 0, 0)))
#         else:
#             # ADD A SOLID BLOCK FOR EACH BIN
#             for i in range(len(alg.ranges) - 1):
#
#                 # Block dimensions
#                 x_min = alg.ranges[i]
#                 x_max = alg.ranges[i + 1]
#
#                 # Alter for loglog
#                 if loglog:
#                     x_min = np.log2(x_min)
#                     x_max = np.log2(x_max)
#
#                 # Add the patch
#                 ax.add_patch(Rectangle(xy=(x_min, 0), width=x_max - x_min, height=max(y), alpha=0.5,
#                                        color=(1, 0, 0) if i % 2 == 0 else (0, 0, 0)))
#
#         if alg_name == EQUAL_F:
#             # Equal-F uses a cutoff, add this to the plot
#             x_co = alg.cutoff
#             if loglog:
#                 x_co = np.log2(x_co)
#
#             ax.plot([x_co] * 2, [min(y), max(y)], 'y--')
#
#         if add_normal_overlay:
#             # Part of the probabilistic membership idea
#             for i in range(alg.number_of_bins):
#                 norm_x = np.linspace(min(x), max(x), num=10000)
#                 norm_y = pdfs[i].pdf(norm_x)
#                 norm_y = norm_y / max(norm_y) * max(y)
#                 ax.plot(norm_x, norm_y, 'w:')
#
#         # Add title for the algorithm
#         ax.set_title(str(alg))
#
#         # Show the plot
#         plt.show()
#
#
# if __name__ == '__main__':
#     plt.close()
#     create_plot_with_all_discretizers(loglog=False, fuzzy_classes=True, add_normal_overlay=True)
#     create_plot_with_all_discretizers(loglog=False, fuzzy_classes=False, add_normal_overlay=True)
#     create_plot_with_all_discretizers(loglog=False, fuzzy_classes=False, add_normal_overlay=False)
