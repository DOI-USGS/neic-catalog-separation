"""Render focal mechanism "beachball" diagrams from strike/dip/rake.

Uses the standard P-wave first-motion radiation pattern formula (Aki, K. &
Richards, P. G. (2002), Quantitative Seismology, 2nd ed., eq. 4.84) to
determine, for every direction on the lower focal hemisphere, whether the
P-wave first motion is compressional (filled) or dilatational (white).
This is the same physical basis used by beachball plotting tools such as
ObsPy's obspy.imaging.beachball, but implemented directly here (matplotlib
only) to avoid adding ObsPy as a dependency. Written with AI-assisted coding
tool Kiro CLI 2.16.2 using claude-sonnet-5.
"""
import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def p_wave_polarity(
    strike: float, dip: float, rake: float, takeoff: np.ndarray, azimuth: np.ndarray
) -> np.ndarray:
    """Compute the (unnormalized) P-wave first-motion radiation amplitude.

    Aki & Richards (2002), eq. 4.84. Positive values are compressional
    (first motion up/away from source); negative are dilatational.

    Args:
        strike, dip, rake: Strike, dip, and rake (degrees) of the first nodal plane.
        takeoff: Takeoff angle(s) (degrees) measured from straight down (0 = down, 90 = horizontal).
        azimuth: Azimuth(s) (degrees) measured clockwise from north.
    Returns:
        Array (same shape as takeoff/azimuth) of radiation amplitudes.
    """
    d2r = np.pi / 180.0
    delta = dip * d2r
    lam = rake * d2r
    ih = takeoff * d2r
    az_phi = (azimuth - strike) * d2r

    term1 = np.cos(lam) * np.sin(delta) * np.sin(ih) ** 2 * np.sin(2.0 * az_phi)
    term2 = -np.cos(lam) * np.cos(delta) * np.sin(2.0 * ih) * np.cos(az_phi)
    term3 = np.sin(lam) * np.sin(2.0 * delta) * (
        np.cos(ih) ** 2 - np.sin(ih) ** 2 * np.sin(az_phi) ** 2
    )
    term4 = np.sin(lam) * np.cos(2.0 * delta) * np.sin(2.0 * ih) * np.sin(az_phi)
    return term1 + term2 + term3 + term4


def focal_mechanism_quadrants(strike: float, dip: float, rake: float, resolution: int = 181):
    """Compute the compressional-quadrant mask over a lower-hemisphere,
    equal-area projection grid for a double-couple focal mechanism.

    Args:
        strike, dip, rake: Strike, dip, and rake (degrees) of the first nodal plane.
        resolution: Number of pixels along each edge of the output grid.
    Returns:
        Tuple (x, y, mask) where x, y are 1-D pixel-center coordinates in
        [-1, 1] and mask is a 2-D boolean array, True where the point is in a
        compressional (filled) quadrant.
    """
    x = np.linspace(-1.0, 1.0, resolution)
    y = np.linspace(-1.0, 1.0, resolution)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    inside = r <= 1.0

    # Lower-hemisphere equal-area (Schmidt) projection: r = sqrt(2) * sin(takeoff/2),
    # where takeoff is measured from straight down (the center of the plot).
    r_clip = np.clip(r, 0.0, 1.0)
    takeoff = np.degrees(2.0 * np.arcsin(r_clip / np.sqrt(2.0)))
    # Azimuth measured clockwise from north; y-axis (up in the plot) is north,
    # x-axis (right in the plot) is east.
    azimuth = np.degrees(np.arctan2(xx, yy))

    amplitude = p_wave_polarity(strike, dip, rake, takeoff, azimuth)
    compressional = (amplitude >= 0.0) & inside

    return x, y, compressional


def render_beachball_png(
    strike: float,
    dip: float,
    rake: float,
    size_px: int = 60,
    compressional_color: str = "black",
    background_color: str = "white",
    edge_color: str = "black",
) -> bytes:
    """Render a focal mechanism beachball as a PNG image.

    Args:
        strike, dip, rake: Strike, dip, and rake (degrees) of the first nodal plane.
        size_px: Output image size (pixels, square).
        compressional_color: Fill color for compressional (filled) quadrants.
        background_color: Fill color for dilatational quadrants.
        edge_color: Color of the outer circle and nodal plane traces.
    Returns:
        PNG-encoded image bytes with a transparent background outside the circle.
    """
    matplotlib.use("Agg")

    resolution = max(size_px, 121)
    x, y, compressional = focal_mechanism_quadrants(strike, dip, rake, resolution=resolution)

    fig = plt.figure(figsize=(size_px / 100.0, size_px / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    xx, yy = np.meshgrid(x, y)
    inside = np.sqrt(xx**2 + yy**2) <= 1.0

    rgba = np.zeros((resolution, resolution, 4))
    bg_rgba = matplotlib.colors.to_rgba(background_color)
    fg_rgba = matplotlib.colors.to_rgba(compressional_color)
    rgba[inside] = bg_rgba
    rgba[compressional] = fg_rgba
    rgba[~inside] = (0.0, 0.0, 0.0, 0.0)  # transparent outside circle

    ax.imshow(rgba, extent=(-1, 1, -1, 1), origin="lower", interpolation="nearest")

    circle = plt.Circle((0, 0), 1.0, fill=False, edgecolor=edge_color, linewidth=1.2)
    ax.add_patch(circle)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", transparent=True)
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()
