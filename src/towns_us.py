"""Flat-land US metro-fringe windows for the urban expert.

The training set so far has zero US windows, so the model grows American
towns like European ones. These sit on 1980-2020 growth frontiers of
flat or gently rolling US metros in the 100k-1M range: gridiron and
subdivision morphology, arterial-strip commercial, low slope variance.
Same rules as towns_urban.py: coordinates are approximate frontier
locations and the growth filter drops windows that did not urbanize.

Usage:  from towns_us import US_TOWNS
        build_dataset_v3(URBAN_TOWNS + US_TOWNS, "data/ds_urban.npz", ...)
"""

US_TOWNS = [
    # --- Midwest: canonical flat gridiron + subdivision growth ---
    ("Des Moines NW fringe", "US", 41.6520, -93.7800, "us_flat"),
    ("Omaha W fringe", "US", 41.2550, -96.1600, "us_flat"),
    ("Wichita NE fringe", "US", 37.7550, -97.2400, "us_flat"),
    ("Indianapolis N fringe", "US", 39.9350, -86.1400, "us_flat"),
    ("Columbus OH NW fringe", "US", 40.1300, -83.0900, "us_flat"),
    ("Madison E fringe", "US", 43.1050, -89.2600, "us_flat"),
    ("Fargo S fringe", "US", 46.8100, -96.8400, "us_flat"),
    ("Sioux Falls S fringe", "US", 43.4900, -96.7300, "us_flat"),
    # --- South: fast 1990-2020 sunbelt expansion ---
    ("Oklahoma City NW fringe", "US", 35.5900, -97.6400, "us_flat"),
    ("Tulsa S fringe", "US", 35.9800, -95.8800, "us_flat"),
    ("Lubbock SW fringe", "US", 33.5200, -101.9300, "us_flat"),
    ("McAllen N fringe", "US", 26.2700, -98.2400, "us_flat"),
    ("Baton Rouge SE fringe", "US", 30.3700, -91.0300, "us_flat"),
    ("Huntsville SE fringe", "US", 34.6700, -86.5300, "us_flat"),
    ("Raleigh NE fringe", "US", 35.8700, -78.5500, "us_flat"),
    ("Lexington S fringe", "US", 37.9700, -84.5100, "us_flat"),
    # --- West: flat basins with sharp growth frontiers ---
    ("Fresno N fringe", "US", 36.8500, -119.7600, "us_flat"),
    ("Boise SW fringe", "US", 43.5700, -116.2900, "us_flat"),
    ("Fort Collins SE fringe", "US", 40.5300, -105.0100, "us_flat"),
    ("Phoenix-Buckeye fringe", "US", 33.4400, -112.4600, "us_flat"),
]
