"""Window lists for the three new experts, plus flat-suburb additions.

The v4 model knew two growth styles: hilly European towns and planned
flat fringes. That misses most of how the world actually grows. These
lists extend coverage in three directions the old pair could not fake:

VILLAGES        settlements small enough that the 1.9 km window is mostly
                empty. Picked from villages that demonstrably grew between
                1980 and 2020 (tourism booms, road access, periurban
                spillover), because the growth filter drops windows with
                no growth to learn from. A village that stayed frozen
                teaches nothing.
INFORMAL_TOWNS  unplanned periurban frontiers: Lima's cones, Nairobi's
                eastlands, Ikorodu, Savar, Orangi. The signature is dense
                building with thin mapped road networks, which is exactly
                the ratio the router keys on. OSM coverage of informal
                areas is uneven; the growth filter eats the bad windows.
MEGACITY_TOWNS  saturated growth edges of the world's largest cities,
                Delhi first among them. Four Delhi-region windows train;
                a fifth (Narela) is held out for eval. Chinese windows
                are included but flagged: OSM building coverage there is
                thin, so expect the filter to drop some.
AUS_NZ_CA_TOWNS flat planned suburbs from Australia, New Zealand, and
                Canada. Morphologically these are US subdivisions with
                different tree species, so they join the URBAN expert's
                training list rather than getting an expert of their own.

Entries follow the towns.py format: (name, country, lat, lon, region).
Coordinates are approximate frontier locations, same contract as
towns_urban.py: imprecision costs samples, not correctness.

Usage:  from towns_global import VILLAGES, INFORMAL_TOWNS, \
            MEGACITY_TOWNS, AUS_NZ_CA_TOWNS
        build_dataset_v3(VILLAGES, "data/ds_village.npz", ...)
"""

VILLAGES = [
    # --- Alpine / Caucasus: tourism-driven village growth ---
    ("Grindelwald", "CH", 46.6240, 8.0340, "village"),
    ("Morzine", "FR", 46.1790, 6.7090, "village"),
    ("Soelden", "AT", 46.9650, 11.0070, "village"),
    ("Ortisei", "IT", 46.5760, 11.6720, "village"),
    ("Mestia", "GE", 43.0450, 42.7270, "village"),
    ("Stepantsminda", "GE", 42.6570, 44.6420, "village"),
    # --- South Asia: hill villages that boomed ---
    ("Vashisht-Manali", "IN", 32.2660, 77.1880, "village"),
    ("Bir", "IN", 32.0330, 76.7200, "village"),
    ("Landour", "IN", 30.4650, 78.1030, "village"),
    ("Besisahar", "NP", 28.2300, 84.3750, "village"),
    # --- SE Asia: backpacker-route villages, explosive growth ---
    ("Pai", "TH", 19.3590, 98.4400, "village"),
    ("Vang Vieng", "LA", 18.9230, 102.4480, "village"),
    ("Canggu", "ID", -8.6380, 115.1380, "village"),
    ("Sidemen", "ID", -8.4660, 115.4280, "village"),
    ("El Nido", "PH", 11.1800, 119.3910, "village"),
    # --- Africa ---
    ("Imlil", "MA", 31.1370, -7.9190, "village"),
    ("Lalibela", "ET", 12.0320, 39.0410, "village"),
    ("Karatu", "TZ", -3.3400, 35.6630, "village"),
    # --- Latin America ---
    ("Salento", "CO", 4.6370, -75.5700, "village"),
    ("Vilcabamba", "EC", -4.2610, -79.2220, "village"),
    ("Bacalar", "MX", 18.6770, -88.3950, "village"),
    ("Samaipata", "BO", -18.1810, -63.8740, "village"),
]

INFORMAL_TOWNS = [
    # --- Latin America ---
    ("Lima N Carabayllo", "PE", -11.8500, -77.0300, "informal"),
    ("Lima S Villa El Salvador fringe", "PE", -12.2400, -76.9300, "informal"),
    ("Bogota S Ciudad Bolivar fringe", "CO", 4.5500, -74.1600, "informal"),
    ("Medellin NE fringe", "CO", 6.3100, -75.5400, "informal"),
    ("Valle de Chalco", "MX", 19.3000, -98.9400, "informal"),
    ("Guatemala City S fringe", "GT", 14.5200, -90.5500, "informal"),
    ("Sao Paulo E Cidade Tiradentes", "BR", -23.6000, -46.4000, "informal"),
    ("Queimados", "BR", -22.7200, -43.5800, "informal"),
    ("Port-au-Prince Canaan", "HT", 18.6200, -72.2700, "informal"),
    # --- Africa ---
    ("Nairobi E Ruai", "KE", -1.2800, 37.0300, "informal"),
    ("Kampala N Kawempe fringe", "UG", 0.4000, 32.5600, "informal"),
    ("Dar es Salaam W Kimara", "TZ", -6.7700, 39.1300, "informal"),
    ("Lagos N Ikorodu", "NG", 6.6200, 3.5100, "informal"),
    ("Ibadan NE fringe", "NG", 7.4400, 3.9500, "informal"),
    ("Ashaiman", "GH", 5.7000, -0.0400, "informal"),
    ("Luanda Viana", "AO", -8.9000, 13.3700, "informal"),
    ("Kinshasa E Nsele", "CD", -4.4000, 15.5200, "informal"),
    ("Burayu", "ET", 9.0500, 38.6500, "informal"),
    ("Shubra el-Kheima N", "EG", 30.1600, 31.2500, "informal"),
    # --- South / SE Asia ---
    ("Savar", "BD", 23.8500, 90.2600, "informal"),
    ("Karachi NW Orangi fringe", "PK", 24.9700, 66.9800, "informal"),
    ("Rodriguez Rizal", "PH", 14.7300, 121.1400, "informal"),
    ("Bekasi-Tambun", "ID", -6.2600, 107.0600, "informal"),
    ("Phnom Penh S fringe", "KH", 11.4900, 104.9000, "informal"),
]

MEGACITY_TOWNS = [
    # --- Delhi region: four training windows, Narela held out ---
    ("Delhi W Najafgarh", "IN", 28.6100, 76.9800, "megacity"),
    ("Delhi E Loni", "IN", 28.7500, 77.2800, "megacity"),
    ("Greater Noida W", "IN", 28.6000, 77.4300, "megacity"),
    ("Gurugram Sohna corridor", "IN", 28.3500, 77.0500, "megacity"),
    # --- Rest of India ---
    ("Vasai-Virar", "IN", 19.3900, 72.8400, "megacity"),
    ("Panvel", "IN", 18.9900, 73.1200, "megacity"),
    ("Kolkata Rajarhat", "IN", 22.6200, 88.4600, "megacity"),
    ("Bengaluru Whitefield fringe", "IN", 12.9900, 77.7600, "megacity"),
    ("Hyderabad Tellapur", "IN", 17.4700, 78.2900, "megacity"),
    ("Chennai OMR corridor", "IN", 12.8700, 80.2300, "megacity"),
    # --- Rest of Asia ---
    ("Dhaka Purbachal", "BD", 23.8200, 90.5200, "megacity"),
    ("Lahore SE fringe", "PK", 31.4200, 74.4400, "megacity"),
    ("Depok fringe", "ID", -6.4200, 106.7900, "megacity"),
    ("Pathum Thani", "TH", 14.0500, 100.6000, "megacity"),
    ("Thu Duc frontier", "VN", 10.8400, 106.8300, "megacity"),
    ("Imus Cavite", "PH", 14.4100, 120.9400, "megacity"),
    ("Guangzhou-Foshan fringe", "CN", 23.0200, 113.0600, "megacity"),
    ("Dongguan W fringe", "CN", 22.8000, 113.8300, "megacity"),
    # --- MENA / Europe edge ---
    ("Istanbul Basaksehir", "TR", 41.0900, 28.8000, "megacity"),
    ("New Cairo edge", "EG", 30.0200, 31.4900, "megacity"),
    ("Shahriar", "IR", 35.6600, 51.0600, "megacity"),
    # --- Americas / Africa ---
    ("Sao Paulo W Cotia fringe", "BR", -23.6000, -46.8500, "megacity"),
    ("Tecamac", "MX", 19.7100, -99.0200, "megacity"),
    ("Lagos Lekki E", "NG", 6.4400, 3.5600, "megacity"),
]

AUS_NZ_CA_TOWNS = [
    ("Melbourne W Tarneit", "AU", -37.8300, 144.6600, "urban_fringe"),
    ("Sydney NW Marsden Park", "AU", -33.6900, 150.8300, "urban_fringe"),
    ("Brisbane S Logan Reserve", "AU", -27.7100, 153.0600, "urban_fringe"),
    ("Perth NE Ellenbrook", "AU", -31.7700, 115.9700, "urban_fringe"),
    ("Adelaide N Munno Para", "AU", -34.6700, 138.7000, "urban_fringe"),
    ("Auckland Flat Bush", "NZ", -36.9700, 174.9200, "urban_fringe"),
    ("Christchurch Halswell", "NZ", -43.5900, 172.5600, "urban_fringe"),
    ("Calgary NE fringe", "CA", 51.1300, -113.9200, "urban_fringe"),
    ("Edmonton SW fringe", "CA", 53.4200, -113.6200, "urban_fringe"),
    ("Winnipeg S fringe", "CA", 49.7800, -97.1500, "urban_fringe"),
]

# Held-out eval windows, one per new expert plus one for the widened
# urban expert. Never used in training.
GLOBAL_EVAL_TOWNS = [
    ("Wengen", "CH", 46.6050, 7.9220, "village"),
    ("Antananarivo W fringe", "MG", -18.9000, 47.4600, "informal"),
    ("Delhi N Narela", "IN", 28.8400, 77.0900, "megacity"),
    ("Melbourne SE Clyde", "AU", -38.1300, 145.3300, "urban_fringe"),
]
