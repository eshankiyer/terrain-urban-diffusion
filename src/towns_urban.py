"""Peri-urban fringe windows for urban-oriented growth training.

Most contemporary settlement growth is not villages expanding but the
edges of mid-size cities absorbing migration. These windows sit on the
1980-2020 growth frontiers of cities in the 100k-1M range (edge
districts, not centres), so the same GHSL temporal pipeline supervises
denser, apartment-scale, infill-heavy growth. Entries follow the
towns.py format: (name, country, lat, lon, region). Coordinates are
approximate frontier locations; the dataset builder's growth filter
drops any window that did not actually urbanize, so imprecision costs
samples rather than correctness.

Usage (mixed training):  from towns import TOWNS
                         from towns_urban import URBAN_TOWNS
                         build_dataset_v3(TOWNS + URBAN_TOWNS, ...)
"""

URBAN_TOWNS = [
    # --- Iberia / France: strong planned-fringe growth ---
    ("Zaragoza SE fringe", "ES", 41.6230, -0.8580, "urban_fringe"),
    ("Valladolid S fringe", "ES", 41.6180, -4.7420, "urban_fringe"),
    ("Braga E fringe", "PT", 41.5560, -8.3900, "urban_fringe"),
    ("Toulouse NE fringe", "FR", 43.6540, 1.4900, "urban_fringe"),
    ("Montpellier W fringe", "FR", 43.6100, 3.8300, "urban_fringe"),
    # --- Central Europe ---
    ("Krakow S fringe", "PL", 50.0130, 19.9500, "urban_fringe"),
    ("Wroclaw W fringe", "PL", 51.1250, 16.9500, "urban_fringe"),
    ("Brno N fringe", "CZ", 49.2280, 16.6100, "urban_fringe"),
    ("Graz S fringe", "AT", 47.0330, 15.4400, "urban_fringe"),
    ("Regensburg E fringe", "DE", 49.0140, 12.1400, "urban_fringe"),
    ("Freiburg N fringe", "DE", 48.0230, 7.8500, "urban_fringe"),
    ("Debrecen W fringe", "HU", 47.5300, 21.5900, "urban_fringe"),
    # --- Balkans / Anatolia: rapid 1990-2020 urbanization ---
    ("Cluj-Napoca S fringe", "RO", 46.7450, 23.5800, "urban_fringe"),
    ("Plovdiv N fringe", "BG", 42.1650, 24.7500, "urban_fringe"),
    ("Novi Sad N fringe", "RS", 45.2800, 19.8300, "urban_fringe"),
    ("Larissa E fringe", "GR", 39.6350, 22.4400, "urban_fringe"),
    ("Bursa E fringe", "TR", 40.2100, 29.1300, "urban_fringe"),
    ("Konya N fringe", "TR", 37.9200, 32.4800, "urban_fringe"),
    ("Denizli W fringe", "TR", 37.7800, 29.0400, "urban_fringe"),
    # --- Asia ---
    ("Chiang Mai S fringe", "TH", 18.7450, 98.9800, "urban_fringe"),
    ("Da Nang S fringe", "VN", 15.9950, 108.2400, "urban_fringe"),
    ("Malang N fringe", "ID", -7.9200, 112.6300, "urban_fringe"),
    ("Cheongju N fringe", "KR", 36.6800, 127.4600, "urban_fringe"),
    ("Chuncheon S fringe", "KR", 37.8400, 127.7400, "urban_fringe"),
    ("Kanazawa S fringe", "JP", 36.5300, 136.6400, "urban_fringe"),
    # --- Africa / MENA ---
    ("Fez S fringe", "MA", 34.0000, -4.9900, "urban_fringe"),
    ("Nakuru E fringe", "KE", -0.2900, 36.1000, "urban_fringe"),
    ("Hawassa N fringe", "ET", 7.0900, 38.5000, "urban_fringe"),
    # --- Americas ---
    ("Arequipa N fringe", "PE", -16.3600, -71.5500, "urban_fringe"),
    ("Bucaramanga S fringe", "CO", 7.0800, -73.1100, "urban_fringe"),
    ("Queretaro N fringe", "MX", 20.6500, -100.4200, "urban_fringe"),
]
