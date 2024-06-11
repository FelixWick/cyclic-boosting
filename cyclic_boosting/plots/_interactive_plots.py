from __future__ import absolute_import, division, print_function

import os
import numpy as np
from plotly import graph_objects as go


# from cyclic_boosting.plots._2dplots import _no_finite_samples
from .plot_utils import blue_cyan_green_cmap, convert_cmap_to_plotly


def _imshow_factors_3d(factors, fname, title, clim, feature):
    """Factor plots for unsmoothed and smoothed 3d factors with interactive slider.
    Parameters
    ----------
    factors
        numpy ndarray (three-dimensional) with factor data
    fname
        file name to save
    title: str
        the title of the plot
    clim: tuple
        limits for imshow
    feature
        Feature object for the feature that is shown
    """
    first_feature_name = feature.feature_group[0] + (feature.feature_type or "")
    second_feature_name = feature.feature_group[1] + (feature.feature_type or "")
    third_feature_name = feature.feature_group[2] + (feature.feature_type or "")

    x_bins = factors.shape[0]
    y_bins = factors.shape[1]
    z_bins = factors.shape[2]
    x = np.arange(0, x_bins)
    y = np.arange(0, y_bins)
    x, y = np.meshgrid(y, x)

    frame = go.Figure()
    colorscale = convert_cmap_to_plotly(blue_cyan_green_cmap())
    heatmap = go.Heatmap(z=factors[:, :, 0].T, zmin=clim[0], zmax=clim[1], colorscale=colorscale)

    frame.add_trace(heatmap)

    # interactive change z
    steps = []
    for i in range(0, z_bins):
        z = factors[:, :, i]
        step = dict(method="update", args=[{"z": [z.T], "visible": True}], label=str(i))
        steps.append(step)

    sliders = go.layout.Slider(
        active=0,
        steps=steps,
        currentvalue={"prefix": third_feature_name + ":", "visible": True},
    )

    frame.update_layout(
        sliders=[sliders], title=title, xaxis={"title": first_feature_name}, yaxis={"title": second_feature_name}
    )

    feature_name = f"({first_feature_name},{second_feature_name},{third_feature_name})"
    frame.write_html(fname + f"{feature_name}.html")


def plot_factor_3d(n_bins_finite, feature):
    """
    Plots a single two dimensional factor plot. For an example see the
    :ref:`cyclic_boosting_analysis_plots`

    Parameters
    ----------
    n_bins_finite: int
        Number of finite bins
    feature: cyclic_boosting.base.Feature
        Feature as it can be obtained from the plotting observers
        ``features`` property.
    """
    from cyclic_boosting.plots import _format_groupname_with_type

    plot_yp = True
    if feature.y is None:
        plot_yp = False
    if plot_yp:
        y2d = feature.y
        prediction2d = feature.prediction
    if plot_yp:
        factors = feature.mean_dev
    else:
        factors = feature.unfitted_factors_link

    smoothed_factors = feature.factors_link
    _ = _format_groupname_with_type(feature.feature_group, feature.feature_type)

    def extremal_factor(x):
        return max(np.abs(np.max(x)), np.abs(np.min(x)))

    factors3d = np.reshape(factors[:-1], n_bins_finite)
    smoothed3d = np.reshape(smoothed_factors[:-1], n_bins_finite)
    if plot_yp:
        y2d = np.reshape(y2d[:-1], n_bins_finite)
        prediction2d = np.reshape(prediction2d[:-1], n_bins_finite)

    if np.prod(n_bins_finite) == 0:
        extremal_absolute_factor = 1
    else:
        extremal_absolute_factor = extremal_factor(smoothed3d)

    clim = (-extremal_absolute_factor, extremal_absolute_factor)

    dir_name = "./interactive"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    _imshow_factors_3d(
        factors3d,
        title="final deviation of predition and truth in link space",
        fname=os.path.join(dir_name, "3d_deviation"),
        clim=clim,
        feature=feature,
    )

    _imshow_factors_3d(
        smoothed3d,
        title="smoothed parameters in link space",
        fname=os.path.join(dir_name, "3d_smoothed_paramaters"),
        clim=clim,
        feature=feature,
    )


def plot_contribution(contributions, X):
    """_summary_

    Args:
        contributions (_type_): _description_
    """
    n_sample = X.shape[0]
    var_names = [name for name in contributions]

    def pick(x, var_names, index):
        contrib = []
        for var in var_names:
            contrib.append(x[var][index,])
        return contrib

    def pick_observation(x, var_names, index):
        observation = []
        src = X.iloc[index,]
        for joint_var in var_names:
            var = joint_var.split(" ")
            if len(var) == 1:
                observation.append(src[var])
            else:
                interaction_term = ""
                interaction_term = interaction_term.join(f"{src[v]}, " for v in var)
                observation.append(interaction_term)

        return observation

    frame = go.Figure()
    ymin, ymax = None, None
    for i in range(0, n_sample):
        contrib = pick(contributions, var_names, i)
        customdata = pick_observation(X, var_names, i)
        scatter = go.Scatter(
            x=var_names,
            y=contrib,
            mode="lines+markers",
            visible=(i == 0),
            customdata=customdata,
            hovertemplate=("%{x}: %{y}<br>" + "value: %{customdata}<br>" + "<extra></extra>"),
        )
        frame.add_trace(scatter)
        curr_min = np.min(contrib)
        curr_max = np.max(contrib)
        ymin = curr_min if (ymin is None) or (curr_min < ymin) else ymin
        ymax = curr_max if (ymax is None) or (curr_max > ymax) else ymax

    # interactive change sample
    steps = []
    for i in range(0, n_sample):
        step = {
            "method": "update",
            "args": [{"visible": [j == i for j in range(0, n_sample)]}],
            "label": str(i),
        }
        steps.append(step)

    sliders = go.layout.Slider(
        active=0,
        steps=steps,
        currentvalue={"prefix": "sample: ", "visible": True},
        pad={"t": 100},
    )

    frame.update_layout(
        sliders=[sliders],
        title="Individual contributions",
        xaxis={"title": "variable"},
        yaxis={"title": "factor", "range": [ymin, ymax]},
    )

    frame.write_html("contribution.html")
