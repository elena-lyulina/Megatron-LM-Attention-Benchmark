"""Offset x prefix x suffix 3D surface / 2D contour panels -- the two visualizations that
feed the memorization dashboard (dashboard/export_data.py). Everything here traces back to
notebooks/mem_plotting_style.ipynb's "add pretty 3d plots" work; earlier tries that never
made it to the dashboard stay in that notebook as scratch."""

import math

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import HTML, IFrame, display
from plotly.subplots import make_subplots

from attn_bench.plotting.data_loading import load_offset_prefix_grid_data

# editorial palette/typography for the plotly 3D surface + 2D contour panels
SERIF = "Georgia, 'Times New Roman', Times, serif"
SANS = "-apple-system, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
INK = '#26241f'
MUTED = '#8a8579'
PANE_BG = 'rgb(250, 249, 245)'
GRID_COLOR = 'rgb(230, 226, 216)'
WALL_FILL = '#c9c4b6'
PAPER_BG = '#fefdfa'
COLORSCALE = px.colors.sequential.thermal
AXIS_FONT = dict(family=SANS, size=11, color=MUTED)
TICK_FONT = dict(family=SANS, size=9, color=MUTED)


def show_plotly(fig, in_browser=False, iframe=False):
    """Default: inline HTML embed (sidesteps renderer/frontend detection issues).
    in_browser=True: temp file opened in the system browser.
    iframe=True: real <iframe> file -- sidesteps a hover-position bug from container CSS
    transforms. uuid-named per call, not plotly's own iframe renderer (which names by cell
    execution count and collides when a cell calls fig.show() more than once)."""
    if in_browser:
        import tempfile
        import webbrowser
        path = tempfile.NamedTemporaryFile(suffix='.html', delete=False).name
        fig.write_html(path)
        webbrowser.open(f'file://{path}')
        return
    if iframe:
        import uuid
        from pathlib import Path
        out_dir = Path('output/iframe_figures')
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f'{uuid.uuid4().hex}.html'
        fig.write_html(path, include_plotlyjs=True)
        display(IFrame(src=str(path), width=(fig.layout.width or 1000) + 20,
                       height=(fig.layout.height or 600) + 20))
        return
    display(HTML(fig.to_html(include_plotlyjs='cdn', full_html=False, config={'responsive': False})))


def _add_surface_panel(fig, row, col, model, rep, suffix, backends=('hf', 'megatron'), metric='Rouge-L',
                       interp='linear', grid_res=80, max_doc_length=8192, floor=-0.2, points=None,
                       show_colorbar=True):
    """Adds one editorial smoothed-surface + document-boundary-wall panel to `fig` at (row,
    col). Shared by plot_offset_prefix_grid_3d's single- and multi-model cases.

    Returns (scene_kwargs, n_points) -- scene_kwargs for the caller's scene/sceneN layout
    key, n_points for titles."""
    offset_vals, prefix_vals, zs, G_OFFSET, G_PREFIX, GZ, feasible_bound = load_offset_prefix_grid_data(
        model, rep, suffix, backends, metric, interp, grid_res, max_doc_length, points)

    hover_template = f'offset: %{{x}}<br>prefix: %{{y}}<br>{metric}: %{{z:.3f}}<extra></extra>'

    # floor shadow: GZ flattened onto a plane below the surface, colored by metric
    floor_z = np.where(np.isnan(GZ), np.nan, floor)
    fig.add_trace(go.Surface(x=G_OFFSET, y=G_PREFIX, z=floor_z, surfacecolor=GZ,
                             colorscale=COLORSCALE, cmin=0, cmax=1, opacity=0.45, showscale=False,
                             hoverinfo='skip'),
                 row=row, col=col)
    fig.add_trace(go.Surface(x=G_OFFSET, y=G_PREFIX, z=GZ, colorscale=COLORSCALE, cmin=0, cmax=1,
                             opacity=0.96, showscale=show_colorbar, hovertemplate=hover_template,
                             lighting=dict(ambient=0.7, diffuse=0.5, specular=0.15, roughness=0.6, fresnel=0.05),
                             lightposition=dict(x=100, y=200, z=300),
                             colorbar=dict(
                                 title=dict(text=metric, font=dict(family=SANS, size=13, color=INK)),
                                 x=1.02, thickness=14, len=0.65,
                                 outlinewidth=0, tickfont=TICK_FONT,
                             )),
                 row=row, col=col)
    fig.add_trace(go.Scatter3d(x=offset_vals, y=prefix_vals, z=zs, mode='markers',
                               marker=dict(size=4, color=zs, colorscale=COLORSCALE, cmin=0, cmax=1,
                                          line=dict(width=0.6, color=INK), opacity=0.95, showscale=False),
                               hovertemplate=hover_template),
                 row=row, col=col)

    # document-boundary wall: vertical plane along offset+prefix=feasible_bound
    wall_offset = np.linspace(0, feasible_bound, grid_res)
    wall_prefix = feasible_bound - wall_offset
    wall_x = np.tile(wall_offset, (2, 1))
    wall_y = np.tile(wall_prefix, (2, 1))
    wall_z = np.tile(np.linspace(floor, 1, 2).reshape(-1, 1), (1, grid_res))
    fig.add_trace(go.Surface(x=wall_x, y=wall_y, z=wall_z, showscale=False, opacity=0.14,
                             colorscale=[[0, WALL_FILL], [1, WALL_FILL]], hoverinfo='skip'),
                 row=row, col=col)

    # z=0.98 not >1 -- zaxis range is fixed to [floor, 1], anything above is clipped and vanishes
    mid_idx = grid_res // 2
    fig.add_trace(go.Scatter3d(x=[wall_x[0, mid_idx]], y=[wall_y[0, mid_idx]], z=[0.98], mode='text',
                               text=[f'document boundary: {max_doc_length} tokens'],
                               textposition='top center',
                               textfont=dict(size=11, color=MUTED, family=SANS),
                               showlegend=False, hoverinfo='skip'),
                 row=row, col=col)

    for zt in (0.0, 0.5, 1.0):
        fig.add_trace(go.Scatter3d(x=wall_x[0], y=wall_y[0], z=[zt] * grid_res, mode='lines',
                                   line=dict(color=GRID_COLOR, width=1.5), opacity=0.7,
                                   showlegend=False, hoverinfo='skip'),
                     row=row, col=col)
    for t in np.linspace(0, 1, 6):
        idx = int(round(t * (grid_res - 1)))
        fig.add_trace(go.Scatter3d(x=[wall_x[0, idx]] * 2, y=[wall_y[0, idx]] * 2, z=[floor, 1],
                                   mode='lines', line=dict(color=GRID_COLOR, width=1.5), opacity=0.7,
                                   showlegend=False, hoverinfo='skip'),
                     row=row, col=col)

    def axis_kwargs(title, extra=None):
        d = dict(
            title=dict(text=title, font=AXIS_FONT),
            tickfont=TICK_FONT,
            gridcolor=GRID_COLOR,
            showbackground=True,
            backgroundcolor=PANE_BG,
            zerolinecolor=GRID_COLOR,
            showspikes=False,
        )
        if extra:
            d.update(extra)
        return d

    scene_kwargs = dict(
        xaxis=axis_kwargs('offset'),
        yaxis=axis_kwargs('prefix'),
        zaxis=axis_kwargs(metric, extra=dict(range=[floor, 1], showbackground=False)),
        camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9)),
    )
    return scene_kwargs, len(offset_vals)


def plot_offset_prefix_grid_3d(models, rep, suffix, ncols=3, backends=('hf', 'megatron'), metric='Rouge-L',
                             interp='linear', grid_res=80, max_doc_length=8192, floor=-0.2, points=None,
                             panel_width=650, panel_height=600):
    """Editorial smoothed surface, one panel per model in `models` (side by side, ncols per
    row). Only the first panel keeps its colorbar. Needs `pip install plotly` (`kaleido`
    too for static PNG export)."""
    n = len(models)
    nrows = math.ceil(n / ncols)
    specs = [[{'type': 'scene'} for _ in range(ncols)] for _ in range(nrows)]
    fig = make_subplots(rows=nrows, cols=ncols, specs=specs, subplot_titles=list(models))

    scene_layout = {}
    for i, model in enumerate(models):
        row, col = i // ncols + 1, i % ncols + 1
        scene_kwargs, _ = _add_surface_panel(fig, row, col, model, rep, suffix, backends, metric,
                                             interp, grid_res, max_doc_length, floor, points,
                                             show_colorbar=(i == 0))
        # make_subplots numbers scene subplots in row-major order over the full grid
        scene_num = (row - 1) * ncols + col
        scene_layout['scene' if scene_num == 1 else f'scene{scene_num}'] = scene_kwargs

    fig.update_layout(
        title=dict(
            text=f'<span style="font-family:{SERIF};font-size:20px;color:{INK}">'
                 f'{metric} at rep={rep}, suffix={suffix}</span>'
                 f'<br><span style="font-size:12px;color:{MUTED}">{interp} interpolation</span>',
            font=dict(family=SANS, size=14, color=INK),
            x=0.02, xanchor='left',
        ),
        font=dict(family=SANS, color=INK),
        paper_bgcolor=PAPER_BG,
        hoverlabel=dict(bgcolor=PANE_BG, bordercolor=GRID_COLOR, font=TICK_FONT),
        width=panel_width * ncols, height=panel_height * nrows, showlegend=False,
        margin=dict(l=10, r=10, t=110, b=10),
        **scene_layout,
    )
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(family=SANS, size=13, color=MUTED)
    show_plotly(fig, iframe=True)


def _add_contour_panel(fig, row, col, model, rep, suffix, backends=('hf', 'megatron'), metric='Rouge-L',
                       interp='linear', grid_res=80, max_doc_length=8192, points=None,
                       show_colorbar=True, show_xlabel=True, show_ylabel=True):
    """2D analog of _add_surface_panel -- filled contour instead of a 3D surface, dashed
    boundary line instead of a wall. show_xlabel/show_ylabel: only True on the bottom row /
    leftmost column.

    Returns n_points, the real-point count, for titles."""
    offset_vals, prefix_vals, zs, G_OFFSET, G_PREFIX, GZ, feasible_bound = load_offset_prefix_grid_data(
        model, rep, suffix, backends, metric, interp, grid_res, max_doc_length, points)

    contour_hover = f'offset: %{{x}}<br>prefix: %{{y}}<br>{metric}: %{{z:.3f}}<extra></extra>'
    # go.Scatter (2D) has no z channel -- %{z} in a hovertemplate silently renders as '-'
    marker_hover = f'offset: %{{x}}<br>prefix: %{{y}}<br>{metric}: %{{customdata:.3f}}<extra></extra>'

    fig.add_trace(go.Contour(x=G_OFFSET[0], y=G_PREFIX[:, 0], z=GZ, colorscale=COLORSCALE,
                             zmin=0, zmax=1, showscale=show_colorbar, connectgaps=False,
                             contours=dict(coloring='fill', showlines=False, start=0, end=1, size=0.01),
                             hovertemplate=contour_hover,
                             colorbar=dict(
                                 title=dict(text=metric, font=dict(family=SANS, size=13, color=MUTED)),
                                 thickness=14, len=0.65, outlinewidth=0, tickfont=TICK_FONT,
                             )),
                 row=row, col=col)
    xref = fig.data[-1].xaxis  # this panel's own x-axis id, for scaleanchor below

    # ring color flips per point (INK above midpoint, GRID_COLOR at/below) to stay legible
    # against both ends of the thermal colorscale
    ring_colors = [INK if z > 0.5 else GRID_COLOR for z in zs]
    fig.add_trace(go.Scatter(x=offset_vals, y=prefix_vals, mode='markers', customdata=zs,
                             marker=dict(size=5, color=zs, colorscale=COLORSCALE, cmin=0, cmax=1,
                                        line=dict(width=0.5, color=ring_colors), opacity=1),
                             hovertemplate=marker_hover, showlegend=False),
                 row=row, col=col)

    fig.add_trace(go.Scatter(x=[0, feasible_bound], y=[feasible_bound, 0], mode='lines',
                             line=dict(color=WALL_FILL, width=1.5, dash='dot'),
                             showlegend=False, hoverinfo='skip'),
                 row=row, col=col)
    # add_annotation (not a text trace) for a rotated label anchored to this panel's own axes
    yref = 'y' + xref[1:]
    mid = feasible_bound / 2
    fig.add_annotation(x=mid, y=mid, xref=xref, yref=yref,
                       text=f'document boundary: {max_doc_length} tokens',
                       textangle=45, showarrow=False, yshift=8,
                       font=dict(size=10, color=MUTED, family=SANS))

    axis_style = dict(gridcolor=GRID_COLOR, tickfont=TICK_FONT, zerolinecolor=GRID_COLOR,
                      showline=True, linecolor=GRID_COLOR, range=[0, max_doc_length])
    fig.update_xaxes(title_text='offset' if show_xlabel else None, title_font=AXIS_FONT,
                     title_standoff=6, row=row, col=col, **axis_style)
    fig.update_yaxes(title_text='prefix' if show_ylabel else None, title_font=AXIS_FONT,
                     title_standoff=6, scaleanchor=xref, scaleratio=1, row=row, col=col, **axis_style)

    return len(offset_vals)


def plot_offset_prefix_grid_2d(models, reps, suffix, backends=('hf', 'megatron'), metric='Rouge-L',
                               interp='linear', grid_res=80, max_doc_length=8192, points=None,
                               panel_size=430, save_path=None):
    """2D analog of plot_offset_prefix_grid_3d: one column per model x one row per repetition.
    save_path: '.html' writes interactive, anything else goes through kaleido as a static
    image."""
    nrows, ncols = len(reps), len(models)
    subplot_titles = [model for i in range(nrows) for model in models]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles,
                        horizontal_spacing=0.03, vertical_spacing=0.045)

    n_points = 0
    for i, rep in enumerate(reps):
        for j, model in enumerate(models):
            n_points += _add_contour_panel(fig, i + 1, j + 1, model, rep, suffix, backends, metric,
                                           interp, grid_res, max_doc_length, points,
                                           show_colorbar=(i == 0 and j == 0),
                                           show_xlabel=(i == nrows - 1), show_ylabel=(j == 0))

    # row labels (repetition) in the left margin -- subplot_titles only labels columns
    for i, rep in enumerate(reps):
        axis_num = i * ncols + 1
        yaxis = fig.layout['yaxis' if axis_num == 1 else f'yaxis{axis_num}']
        y0, y1 = yaxis.domain
        fig.add_annotation(text=f'rep={rep}', x=0, xshift=-48, xanchor='right',
                           y=(y0 + y1) / 2, xref='paper', yref='paper',
                           showarrow=False, textangle=-90, font=dict(family=SANS, size=13, color=MUTED))

    for ann in fig['layout']['annotations']:
        if ann['text'] in models:
            ann['font'] = dict(family=SANS, size=13, color=MUTED)

    avg_points = n_points / (nrows * ncols) if nrows * ncols else 0
    bounds_desc = ''
    if points is not None:
        offset_bounds = ', '.join(str(o) for o in sorted({o for o, p in points}))
        prefix_bounds = ', '.join(str(p) for p in sorted({p for o, p in points}))
        bounds_desc = f'offset: {offset_bounds} · prefix: {prefix_bounds} · '

    fig.update_layout(
        title=dict(
            text=f'<span style="font-size:20px;color:{INK}">{metric} at suffix={suffix}</span>'
                 f'<br><span style="font-size:12px;color:{MUTED}">'
                 f'{interp} interpolation, extrapolated · {bounds_desc}{avg_points:.1f} points/plot avg</span>',
            font=dict(family=SANS, size=14, color=INK),
            x=0.02, xanchor='left',
        ),
        font=dict(family=SANS, color=INK),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PANE_BG,
        hoverlabel=dict(bgcolor=PANE_BG, bordercolor=GRID_COLOR, font=TICK_FONT),
        width=panel_size * ncols, height=panel_size * nrows, showlegend=False,
        margin=dict(l=95, r=10, t=110, b=50),
    )
    if save_path is not None:
        if str(save_path).endswith('.html'):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)
    show_plotly(fig, iframe=True)
