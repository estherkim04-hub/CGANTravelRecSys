import time
import pandas as pd
import numpy as np
import networkx as nx
import math
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from collections.abc import Iterable
import folium
import random

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Embedding as KerasEmbedding, Flatten,
    Dense, Concatenate, LSTM, BatchNormalization, LeakyReLU,
    Dropout, GaussianNoise, Wrapper
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.spatial.distance import cosine, jensenshannon
from itertools import permutations
from scipy.linalg import sqrtm
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance


# -----------------------------------------------
# 1. Environment setup and data loading
# -----------------------------------------------
script_start = time.time()

def normalize_poi_name(name):
    """Convert POI name to a consistent format (lowercase, strip leading/trailing spaces)"""
    return str(name).strip().lower()

# Test out with different check-in dataset from 5 cities.
# ===================================================================================================
#Melbourne
#file_path = '{The path where you put the whole system}/dataset/melb_userRoutes_with_theme_preferences.csv'

#Edinburgh
#file_path = '{The path where you put the whole system}/dataset/toro_finale_userRoutes_with_theme_preferences.csv'

#Glasgow
#file_path = '{The path where you put the whole system}/dataset/glas_finale_userRoutes_with_theme_preferences.csv'

#Toronto
#file_path = '{The path where you put the whole system}/dataset/toro_finale_userRoutes_with_theme_preferences.csv'

#Osaka
#file_path = '{The path where you put the whole system}/dataset/osak_finale_userRoutes_with_theme_preferences.csv'
# ===================================================================================================

df = pd.read_csv(file_path, encoding='utf-8')

df['itemID'] = df.index.astype(str)

moving_cols   = [c for c in df.columns if c.startswith('movingpath') and c[len('movingpath'):].isdigit()]
theme_cols    = [c for c in df.columns if c.startswith('theme')      and c[len('theme'):].isdigit()]
subtheme_cols = [c for c in df.columns if c.startswith('subTheme')   and c[len('subTheme'):].isdigit()]
poiFreq_cols  = [c for c in df.columns if c.startswith('poiFreq')    and c[len('poiFreq'):].isdigit()]
df['totalFreq'] = df[poiFreq_cols].sum(axis=1)

poiID_cols = [c for c in df.columns if c.startswith('poiID') and c[len('poiID'):].isdigit()]
lat_cols   = [c for c in df.columns if c.startswith('lat')   and c[len('lat'):].isdigit()]
long_cols  = [c for c in df.columns if c.startswith('long')  and c[len('long'):].isdigit()]
pref_cols  = [c for c in df.columns if c.startswith('pref_')]

valid_pref_cols = [col for col in pref_cols if col in df.columns]
if not valid_pref_cols:
    theme_list_pref = []
else:
    theme_list_pref = [c.replace('pref_','') for c in valid_pref_cols]

theme_dim_pref = len(theme_list_pref)

# Themes to exclude (which isn't fit for travel)
exclude_themes = {
    'Community Use',
    'Health Services',
    'Education Centre',
    'Office',
    'Residential Accommodation',
    'Mixed Use',
    'Vacant Land',
    'Transport',
    'Specialist Residential Accommodation'
}

# -----------------------------------------------
# 2. Remove POI information corresponding to excluded themes
# -----------------------------------------------
for i in range(len(moving_cols)):
    theme_col = theme_cols[i] if i < len(theme_cols) else None
    if theme_col in df.columns:
        mask = df[theme_col].isin(exclude_themes)
        df.loc[mask, moving_cols[i]] = np.nan
        if i < len(poiID_cols) and poiID_cols[i] in df.columns:
             df.loc[mask, poiID_cols[i]]  = np.nan
        if i < len(lat_cols) and lat_cols[i] in df.columns:
            df.loc[mask, lat_cols[i]]    = np.nan
        if i < len(long_cols) and long_cols[i] in df.columns:
            df.loc[mask, long_cols[i]]   = np.nan
        if i < len(poiFreq_cols) and poiFreq_cols[i] in df.columns:
            df.loc[mask, poiFreq_cols[i]] = np.nan
        if i < len(subtheme_cols) and subtheme_cols[i] in df.columns:
            df.loc[mask, subtheme_cols[i]] = np.nan


for theme in exclude_themes:
    pref_col = f'pref_{theme.replace("/", "_").replace(" ", "_")}' # Ensure space replacement for consistency
    if pref_col in df.columns:
        df[pref_col] = 0.0

# -----------------------------------------------
# 3. valid_places("without excluded themes") set
# -----------------------------------------------
place_themes    = {}
place_subthemes = {}
for _, row in df.iterrows():
    for i, pcol in enumerate(moving_cols):
        place = row[pcol]
        if pd.isnull(place):
            continue
        th  = row[theme_cols[i]] if i < len(theme_cols) and pd.notnull(row[theme_cols[i]]) else None
        sth = row[subtheme_cols[i]] if i < len(subtheme_cols) and pd.notnull(row[subtheme_cols[i]]) else None
        if th:
            place_themes.setdefault(place, set()).add(th)
        if sth:
            place_subthemes.setdefault(place, set()).add(sth)

valid_places = {p for p, themes in place_themes.items() if not themes.issubset(exclude_themes)}
place_themes = {p: themes for p, themes in place_themes.items() if p in valid_places}
place_subthemes = {p: subthemes for p, subthemes in place_subthemes.items() if p in valid_places}

user_visited_places = df.groupby('userID').apply(
    lambda x: set(x[moving_cols].values.ravel()).difference({np.nan}).intersection(valid_places)
).to_dict()

# -----------------------------------------------
# 4. User theme and subtheme profile creation
# -----------------------------------------------
user_theme_profile    = defaultdict(Counter)
user_subtheme_profile = defaultdict(Counter)
for _, row in df.iterrows():
    user_id = row['userID']
    for i, pcol in enumerate(moving_cols):
        place = row[pcol]
        if pd.notnull(place) and (place in valid_places):
            theme    = row[theme_cols[i]]    if i < len(theme_cols)    and pd.notnull(row[theme_cols[i]])    else None
            subtheme = row[subtheme_cols[i]] if i < len(subtheme_cols) and pd.notnull(row[subtheme_cols[i]]) else None
            poi_freq = row[poiFreq_cols[i]]  if i < len(poiFreq_cols) and pd.notnull(row[poiFreq_cols[i]]) else 0 # Handle potential NaN in poiFreq
            
            if theme and theme not in exclude_themes:
                user_theme_profile[user_id][theme] += poi_freq
            if subtheme: # Assuming subthemes don't need explicit exclusion if their parent themes are handled
                user_subtheme_profile[user_id][subtheme] += poi_freq


all_themes_set = set()
for p in valid_places:
    if p in place_themes:
        for t in place_themes[p]:
            if t not in exclude_themes:
                 all_themes_set.add(t)
all_themes = sorted(list(all_themes_set))

all_subthemes_set = set()
for p in valid_places:
    if p in place_subthemes:
        for st in place_subthemes[p]:
            # Add condition if subthemes need filtering based on parent theme exclusion
            all_subthemes_set.add(st)
all_subthemes = sorted(list(all_subthemes_set))


theme_dim_orig = len(all_themes)
subtheme_dim_orig = len(all_subthemes)
theme_index    = {t: i for i, t in enumerate(all_themes)}
subtheme_index = {st: i for i, st in enumerate(all_subthemes)}

# -----------------------------------------------
# 5. User theme and subtheme frequency vector functions
# -----------------------------------------------
def get_user_theme_freq_vector(user_id):
    counter = user_theme_profile.get(user_id, Counter())
    vec = np.zeros(theme_dim_orig)
    total = sum(counter.values())
    for t, count in counter.items():
        if t in theme_index and total > 0: # Ensure theme is in the filtered all_themes
            vec[theme_index[t]] = count / total
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8) if norm > 0 else vec


def get_user_subtheme_freq_vector(user_id):
    counter = user_subtheme_profile.get(user_id, Counter())
    vec = np.zeros(subtheme_dim_orig)
    total = sum(counter.values())
    for st, count in counter.items():
        if st in subtheme_index and total > 0: # Ensure subtheme is in filtered all_subthemes
            vec[subtheme_index[st]] = count / total
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8) if norm > 0 else vec  

# -----------------------------------------------
# 6. Place → (latitude, longitude) mapping generation
# -----------------------------------------------
place_to_coord = {}
for _, row in df.iterrows():
    for i in range(len(moving_cols)):
        mcol = moving_cols[i]
        latcol = lat_cols[i] if i < len(lat_cols) else None
        loncol = long_cols[i] if i < len(long_cols) else None

        place = row[mcol]
        if place not in valid_places or pd.isnull(place):
            continue
        
        if latcol and loncol and latcol in row and loncol in row:
            lat = row[latcol]
            lon = row[loncol]
            if pd.notnull(lat) and pd.notnull(lon):
                try: # Add try-except for robustness if lat/lon are not purely numeric
                    place_to_coord[place] = (float(lat), float(lon))
                except ValueError:
                    # print(f"Warning: Could not convert lat/lon to float for place {place}: {lat}, {lon}")
                    pass


def get_user_geo_center(user_id):
    visited = user_visited_places.get(user_id, [])
    if not visited:
        return None
    coords = [place_to_coord[p] for p in visited if p in place_to_coord]
    if coords:
        return np.mean(coords, axis=0)
    return None

# -----------------------------------------------
# 7. Distance calculation functions
# -----------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    try:
        phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
        dphi = math.radians(float(lat2) - float(lat1))
        dlambda = math.radians(float(lon2) - float(lon1))
    except ValueError:
        # print(f"Warning: Invalid coordinates for Haversine: ({lat1},{lon1}), ({lat2},{lon2})")
        return np.inf # Or some other large number / error indicator
        
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def compute_route_distance(seq, place_to_coord_map, user_id, user_visited_places_map, df_orig, mv_cols):
    if not seq or not all(isinstance(p, str) for p in seq): # Ensure seq contains place names
        return 1.0 
    
    # Find the actual last POI from the user's recorded history in df_orig
    urows = df_orig[df_orig['userID'] == user_id]
    if urows.empty:
        return 1.0 # Default if user not found (should not happen if user_id is valid)

    user_actual_last_poi = None
    for _, u_row in urows.iterrows(): # Iterate through all rows of the user to find the latest valid POI
        for col in reversed(mv_cols):
            last_val_candidate = u_row[col]
            if pd.notnull(last_val_candidate) and last_val_candidate in place_to_coord_map:
                user_actual_last_poi = last_val_candidate
                break # Found last POI in this row
        if user_actual_last_poi:
             break # Found last POI for the user across their rows


    total_distance = 0.0
    current_loc_for_dist_calc = None

    if user_actual_last_poi:
        current_loc_for_dist_calc = place_to_coord_map.get(user_actual_last_poi)
    
    if current_loc_for_dist_calc is None: # Fallback to geo_center if last POI not usable
        geo_center = get_user_geo_center(user_id) # Uses pre-computed user_visited_places
        if geo_center is not None:
            current_loc_for_dist_calc = geo_center

    # Distance from current_loc_for_dist_calc (last known or geo_center) to first POI of new route
    if current_loc_for_dist_calc and seq[0] in place_to_coord_map:
        lat1, lon1 = current_loc_for_dist_calc
        lat2, lon2 = place_to_coord_map[seq[0]]
        total_distance += haversine(lat1, lon1, lat2, lon2)
    
    # Distance within the new route sequence
    for i in range(len(seq) - 1):
        p1, p2 = seq[i], seq[i+1]
        if p1 in place_to_coord_map and p2 in place_to_coord_map:
            lat1, lon1 = place_to_coord_map[p1]
            lat2, lon2 = place_to_coord_map[p2]
            total_distance += haversine(lat1, lon1, lat2, lon2)
        # else:
            # print(f"Warning: Coordinates missing for distance calculation between {p1} and {p2}")

    return max(total_distance, 1.0) # Ensure distance is at least 1.0

# -----------------------------------------------
# 7.5. Confidence based weight and contrastive learning functions (NEW)
# -----------------------------------------------

def calculate_interaction_confidence(user_id, df, moving_cols, poiFreq_cols):
    """Interaction confidence calculation"""
    user_rows = df[df['userID'] == user_id]
    
    confidence_scores = {}
    
    for _, row in user_rows.iterrows():
        for i, mcol in enumerate(moving_cols):
            poi = row[mcol]
            if pd.notnull(poi) and poi in valid_places:
                freq = row[poiFreq_cols[i]] if i < len(poiFreq_cols) else 1
                freq_confidence = min(1.0, freq / 10.0)  # 10회 이상시 최대 신뢰도
                
                poi_visits = []
                for _, other_row in user_rows.iterrows():
                    for j, other_mcol in enumerate(moving_cols):
                        if pd.notnull(other_row[other_mcol]) and other_row[other_mcol] == poi:
                            poi_visits.append(j)  # 방문 순서 기록
                
                temporal_consistency = len(set(poi_visits)) / len(poi_visits) if poi_visits else 0
                

                geo_consistency = calculate_geo_consistency(poi, user_id, user_rows, moving_cols)
                
                confidence_scores[poi] = {
                    'frequency_conf': freq_confidence,
                    'temporal_conf': temporal_consistency,
                    'geo_conf': geo_consistency,
                    'total_conf': (freq_confidence * 0.4 + temporal_consistency * 0.3 + geo_consistency * 0.3)
                }
    
    return confidence_scores

def calculate_geo_consistency(target_poi, user_id, user_rows, moving_cols):
    """Geographical Consistency Calculation"""
    if target_poi not in place_to_coord:
        return 0.5
    
    target_coord = place_to_coord[target_poi]
    nearby_visited = 0
    total_visited = 0
    
    for _, row in user_rows.iterrows():
        for mcol in moving_cols:
            poi = row[mcol]
            if pd.notnull(poi) and poi in place_to_coord and poi != target_poi:
                total_visited += 1
                poi_coord = place_to_coord[poi]
                distance = haversine(target_coord[0], target_coord[1], 
                                   poi_coord[0], poi_coord[1])
                if distance <= 5.0:
                    nearby_visited += 1
    
    return nearby_visited / total_visited if total_visited > 0 else 0.5

def compute_multi_signal_features(user_id, df, moving_cols, confidence_scores):
    """Multi-signal features"""
    user_rows = df[df['userID'] == user_id]
    
    # 1. Theme frequency vector
    theme_freq_v = get_user_theme_freq_vector(user_id)
    subtheme_freq_v = get_user_subtheme_freq_vector(user_id)
    
    geo_center_temp = get_user_geo_center(user_id)
    if geo_center_temp is None:
        geo_center_v = np.zeros(2)
    else:
        geo_center_v = geo_center_temp
    
    # 2. Temporal patterns analyze
    temporal_features = analyze_temporal_patterns(user_id, user_rows, moving_cols)
    
    # 3. Confidence-weighted features
    weighted_features = compute_confidence_weighted_features(user_id, confidence_scores)
    
    # 4. Uncertainty features
    uncertainty_features = compute_uncertainty_features(user_id, user_rows, confidence_scores)
    
    # 5. Visit pattern diversity
    diversity_features = compute_visit_diversity(user_id, user_rows, moving_cols)
    
    return np.concatenate([
        theme_freq_v,
        subtheme_freq_v, 
        geo_center_v,
        temporal_features,
        weighted_features,
        uncertainty_features,
        diversity_features
    ])

def analyze_temporal_patterns(user_id, user_rows, moving_cols):
    theme_transitions = []
    for _, row in user_rows.iterrows():
        row_themes = []
        for mcol in moving_cols:
            poi = row[mcol]
            if pd.notnull(poi) and poi in place_themes:
                row_themes.extend(list(place_themes[poi]))
        
        for i in range(len(row_themes) - 1):
            theme_transitions.append((row_themes[i], row_themes[i+1]))
    
    # Transition stability
    transition_stability = len(set(theme_transitions)) / len(theme_transitions) if theme_transitions else 0
    
    visit_lengths = []
    for _, row in user_rows.iterrows():
        length = sum(1 for mcol in moving_cols if pd.notnull(row[mcol]))
        visit_lengths.append(length)
    
    avg_visit_length = np.mean(visit_lengths) if visit_lengths else 0
    visit_length_variance = np.var(visit_lengths) if visit_lengths else 0
    
    return np.array([transition_stability, avg_visit_length, visit_length_variance])

def compute_confidence_weighted_features(user_id, confidence_scores):
    """Confidence-weighted features"""
    user_confidence = confidence_scores.get(user_id, {})
    
    if not user_confidence:
        return np.array([0.5, 0.5, 0.5, 0.5])
    
    conf_values = [poi_conf['total_conf'] for poi_conf in user_confidence.values()]
    
    avg_confidence = np.mean(conf_values)
    confidence_variance = np.var(conf_values) 
    min_confidence = np.min(conf_values)
    max_confidence = np.max(conf_values)
    
    return np.array([avg_confidence, confidence_variance, min_confidence, max_confidence])

def compute_uncertainty_features(user_id, user_rows, confidence_scores):
    """Uncertainty features calculation"""
    user_confidence = confidence_scores.get(user_id, {})
    
    # 1. Preference uncertainty (based on variance)
    theme_prefs = get_user_theme_freq_vector(user_id)
    preference_uncertainty = np.var(theme_prefs) if len(theme_prefs) > 0 else 1.0
    
    # 2. Data scarcity uncertainty
    total_interactions = len(user_rows)
    data_scarcity_uncertainty = 1.0 / (1.0 + total_interactions * 0.1)  # More interactions = lower uncertainty
    
    # 3. Confidence uncertainty
    if user_confidence:
        conf_values = [poi_conf['total_conf'] for poi_conf in user_confidence.values()]
        confidence_uncertainty = 1.0 - np.mean(conf_values)
    else:
        confidence_uncertainty = 1.0
    
    # 4. Geographical Uncertainty (Variation of visited locations)
    user_coords = []
    for _, row in user_rows.iterrows():
        for mcol in moving_cols:
            poi = row[mcol]
            if pd.notnull(poi) and poi in place_to_coord:
                user_coords.append(place_to_coord[poi])
    
    if len(user_coords) > 1:
        geo_variance = np.var(user_coords, axis=0).mean()
        geo_uncertainty = min(1.0, geo_variance / 100.0)  # Normalization
    else:
        geo_uncertainty = 1.0
    
    return np.array([
        preference_uncertainty,
        data_scarcity_uncertainty, 
        confidence_uncertainty,
        geo_uncertainty
    ])

def compute_visit_diversity(user_id, user_rows, moving_cols):
    """Visit Pattern Diversity"""
    all_themes = set()
    all_pois = set()
    
    for _, row in user_rows.iterrows():
        for mcol in moving_cols:
            poi = row[mcol]
            if pd.notnull(poi):
                all_pois.add(poi)
                if poi in place_themes:
                    all_themes.update(place_themes[poi])
    
    # Theme Diversity
    theme_diversity = len(all_themes) / len(all_themes_set) if all_themes_set else 0
    
    # POI Diversity
    poi_diversity = len(all_pois) / len(valid_places) if valid_places else 0
    
    # Repetitive Visit Ratio
    total_visits = sum(1 for _, row in user_rows.iterrows() 
                      for mcol in moving_cols if pd.notnull(row[mcol]))
    repeat_ratio = 1.0 - (len(all_pois) / total_visits) if total_visits > 0 else 0
    
    return np.array([theme_diversity, poi_diversity, repeat_ratio])    

# -----------------------------------------------
# 8. HIN & Metapath2Vec → User embedding
# -----------------------------------------------
def build_hin(df_hin):
    G = nx.Graph()
    # Add user nodes
    for user_node in df_hin['userID'].unique():
        G.add_node(user_node, node_type='user')
    # Add route nodes
    for route_node in df_hin['itemID'].unique():
        G.add_node(route_node, node_type='route')
    # Add place nodes (only valid places)
    for place_node in valid_places: # Use pre-filtered valid_places
        G.add_node(place_node, node_type='place')

    for _, row in df_hin.iterrows():
        u, r = row['userID'], row['itemID']
        if G.has_node(u) and G.has_node(r): # Ensure nodes exist before adding edge
            G.add_edge(u, r, edge_type='UR')
        
        for col in moving_cols:
            p = row[col]
            if pd.notnull(p) and p in valid_places: # Check if p is a valid place
                 if G.has_node(r) and G.has_node(p): # Ensure nodes exist
                    G.add_edge(r, p, edge_type='RP')
    return G

def generate_metapaths(G, num_walks=10):
    metapath_schemas = {
        'URPRU': ['user', 'route', 'place', 'route', 'user'], # U-R-P-R-U
        'URU':   ['user', 'route', 'user'] # U-R-U
        # Add more metapaths like U-P-U if direct user-place interactions are meaningful and modeled
    }
    
    walks = []
    nodes = list(G.nodes())
    
    for _ in range(num_walks): # Number of walks per starting node (or total walks)
        random.shuffle(nodes) # Process nodes in random order
        for start_node in nodes:
            node_type_start = G.nodes[start_node].get('node_type')
            
            for mp_name, mp_schema in metapath_schemas.items():
                if node_type_start != mp_schema[0]: # Start node must match the beginning of metapath
                    continue

                current_walk = [start_node]
                current_node = start_node
                
                # Try to follow the metapath schema
                possible_path = True
                for i in range(len(mp_schema) - 1):
                    target_node_type = mp_schema[i+1]
                    
                    # Find neighbors of current_node that match target_node_type
                    # And also consider edge_type if defined (e.g., UR, RP)
                    # For simplicity here, we just check node type. Edge type check would be more robust for specific metapaths.
                    
                    valid_neighbors = [
                        n for n in G.neighbors(current_node) 
                        if G.nodes[n].get('node_type') == target_node_type
                    ]
                    
                    # Specific edge type check based on URPRU / URU
                    # Example, if current is 'user' and target is 'route', edge must be 'UR'
                    # This requires G.edges[u,v]['edge_type'] to be consistently set.
                    # The current build_hin sets G[cur][n]['edge_type']

                    if mp_schema[i] == 'user' and mp_schema[i+1] == 'route':
                        edge_t = 'UR'
                        valid_neighbors = [n for n in valid_neighbors if G.has_edge(current_node, n) and G[current_node][n].get('edge_type') == edge_t]
                    elif mp_schema[i] == 'route' and mp_schema[i+1] == 'place':
                        edge_t = 'RP'
                        valid_neighbors = [n for n in valid_neighbors if G.has_edge(current_node, n) and G[current_node][n].get('edge_type') == edge_t]
                    elif mp_schema[i] == 'place' and mp_schema[i+1] == 'route': # Reversed RP for P-R
                        edge_t = 'RP' # Edges are undirected, so (r,p) with RP also means (p,r) with RP
                        valid_neighbors = [n for n in valid_neighbors if G.has_edge(current_node, n) and G[current_node][n].get('edge_type') == edge_t]
                    elif mp_schema[i] == 'route' and mp_schema[i+1] == 'user': # Reversed UR for R-U
                        edge_t = 'UR'
                        valid_neighbors = [n for n in valid_neighbors if G.has_edge(current_node, n) and G[current_node][n].get('edge_type') == edge_t]


                    if not valid_neighbors:
                        possible_path = False
                        break 
                    
                    current_node = random.choice(valid_neighbors)
                    current_walk.append(current_node)
                
                if possible_path and len(current_walk) == len(mp_schema): # Ensure full metapath was traversed
                    walks.append(current_walk)
    return walks


def train_metapath2vec(df_mp2v, emb_size=64, num_walks_per_node=10, epochs=5): # Added epochs
    G = build_hin(df_mp2v)
    print(f"HIN graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        print("Warning: HIN is empty or has no edges. Metapath2Vec will not be effective.")
        # Return dictionary of zero vectors for users
        user_emb_zeros = {u: np.zeros(emb_size) for u in df_mp2v['userID'].unique()}
        # And a dummy Word2Vec model or None
        return user_emb_zeros, None 

    walks = generate_metapaths(G, num_walks=num_walks_per_node) # num_walks is total walks here
    print(f"Generated {len(walks)} metapaths.")

    if not walks:
        print("Warning: No metapaths generated. Metapath2Vec cannot be trained.")
        user_emb_zeros = {u: np.zeros(emb_size) for u in df_mp2v['userID'].unique()}
        return user_emb_zeros, None

    walks_str = [[str(node) for node in w] for w in walks] # Word2Vec expects strings
    
    # Ensure min_count is not too high if dataset/walks are small
    min_c = 1 if len(walks_str) < 50 else 5 

    w2v = Word2Vec(walks_str, vector_size=emb_size, window=5,
                   min_count=min_c, sg=1, epochs=epochs, workers=4) # Added epochs, workers
    
    user_emb = {}
    for u_id in df_mp2v['userID'].unique():
        str_uid = str(u_id)
        if str_uid in w2v.wv:
            user_emb[u_id] = w2v.wv[str_uid]
        else:
            user_emb[u_id] = np.zeros(emb_size) # Fallback for users not in walks
            
    return user_emb, w2v

user_emb_meta, w2v_model = train_metapath2vec(df, emb_size=64, num_walks_per_node=5, epochs=10) # Reduced walks, increased epochs
user_dim_meta = next(iter(user_emb_meta.values()), np.zeros(64)).shape[0]


# -----------------------------------------------
# 9. Route RNN → Latent
# -----------------------------------------------
def train_route_rnn(df_rnn, emb_size=64, max_len=10, epochs=10): # Added epochs
    places_sorted = sorted(list(valid_places)) # Ensure places are from valid_places
    p2i_map = {p: i+1 for i, p in enumerate(places_sorted)} # 0 is for padding
    p2i_map['<PAD>'] = 0 # Explicit padding token

    routes_unique = df_rnn['itemID'].unique()
    seqs = []
    for r_id in routes_unique:
        row = df_rnn[df_rnn['itemID'] == r_id].iloc[0] # Assuming one row per itemID for its definition
        seq = []
        for c in moving_cols:
            p = row[c]
            if pd.notnull(p) and p in p2i_map: # Check if place is valid and in map
                seq.append(p2i_map[p])
        if seq: # Only add non-empty sequences
            seqs.append(seq)
        # else:
            # print(f"Warning: Route {r_id} resulted in empty sequence for RNN.")


    if not seqs:
        print("Error: No valid sequences for RNN training.")
        # Return empty embeddings or handle error
        return {}, p2i_map, None 

    X_rnn = pad_sequences(seqs, maxlen=max_len, padding='pre', value=p2i_map['<PAD>'])
    
    # Model
    # Using len(p2i_map) because p2i_map includes the <PAD> token if its index is highest, or +1 if its 0
    # input_dim should be max_index + 1
    input_dim_rnn = max(p2i_map.values()) + 1

    model_rnn = Sequential([
        KerasEmbedding(input_dim=input_dim_rnn, output_dim=emb_size, input_length=max_len, mask_zero=True), # mask_zero for padding
        LSTM(64), # Can be same as emb_size or different
        Dense(emb_size, activation='linear') # Output matches embedding size
    ])
    model_rnn.compile(optimizer='adam', loss='mse')
    
    # Dummy targets for autoencoder-like training (or use actual targets if available)
    # Here, we predict the embedding itself, so a random target is not ideal.
    # A common approach for sequence autoencoders is to try and reconstruct the input or a compressed version.
    # For simplicity, let's use a placeholder target, but this is not ideal for learning meaningful embeddings.
    # A better approach would be a sequence-to-sequence model or predicting next POI.
    # Given the current setup, this is more like learning a fixed-size representation.
    
    # Using a simple autoencoder target: predict the average embedding of input POIs (conceptual)
    # For now, keeping the random target as in original to avoid major restructuring here.
    # This part can be significantly improved with a more principled RNN objective.
    dummy_targets = np.random.rand(len(X_rnn), emb_size) 
    model_rnn.fit(X_rnn, dummy_targets, epochs=epochs, batch_size=32, verbose=0)
    
    embeddings_rnn = model_rnn.predict(X_rnn, verbose=0)
    
    # Map embeddings back to original route_ids that formed the sequences
    route_emb_map_rnn = {}
    seq_idx = 0
    for r_id in routes_unique: # Iterate in the same order as seqs were created
        row = df_rnn[df_rnn['itemID'] == r_id].iloc[0]
        temp_seq = []
        for c in moving_cols:
            p = row[c]
            if pd.notnull(p) and p in p2i_map:
                temp_seq.append(p2i_map[p])
        if temp_seq: # If this route contributed a sequence
            route_emb_map_rnn[r_id] = embeddings_rnn[seq_idx]
            seq_idx +=1
        else: # Route had no valid POIs for RNN
            route_emb_map_rnn[r_id] = np.zeros(emb_size)


    # Fallback for any routes that might not have been processed if they had no POIs
    for r_id in routes_unique:
        if r_id not in route_emb_map_rnn:
            route_emb_map_rnn[r_id] = np.zeros(emb_size)


    return route_emb_map_rnn, p2i_map, model_rnn

route_emb_rnn, p2i, rnn_model = train_route_rnn(df, emb_size=64, max_len=10, epochs=15)

# Ensure all itemIDs from df are in route_emb_rnn, even if with zeros
all_item_ids_from_df = df['itemID'].unique()
for item_id_val in all_item_ids_from_df:
    if item_id_val not in route_emb_rnn:
        route_emb_rnn[item_id_val] = np.zeros(64) # Default emb_size

route_ids_rnn = list(route_emb_rnn.keys()) # These are the itemIDs
# Check if route_emb_rnn is empty
if not route_ids_rnn:
    print("Error: RNN training resulted in no route embeddings. Exiting or handling.")
    # Handle this case: maybe use zero embeddings or stop.
    # For now, we'll let it proceed but it will likely fail downstream if PCA needs non-empty input.
    route_latents_rnn = np.array([]) # Empty array
    orig_latent_dim = 0
    knn_real = None # Cannot fit KNN
    route_index = {}
else:
    route_mat_rnn = np.vstack([route_emb_rnn[r] for r in route_ids_rnn])
    
    # PCA for RNN embeddings
    # Ensure n_components is less than or equal to min(n_samples, n_features)
    n_components_pca_route = min(64, route_mat_rnn.shape[0], route_mat_rnn.shape[1]) # Adjusted PCA components
    if n_components_pca_route > 0 :
        pca_route = PCA(n_components=n_components_pca_route)
        route_latents_rnn_pca_transformed = pca_route.fit_transform(route_mat_rnn)
    else: # Not enough samples/features for PCA
        print("Warning: Not enough data for PCA on RNN embeddings. Using raw RNN embeddings or zeros.")
        route_latents_rnn_pca_transformed = route_mat_rnn if route_mat_rnn.shape[1] > 0 else np.zeros((route_mat_rnn.shape[0], 1))


    orig_latent_dim = route_latents_rnn_pca_transformed.shape[1]
    if orig_latent_dim == 0 and route_mat_rnn.shape[0] > 0: # If PCA resulted in 0 dim but we have routes
         orig_latent_dim = 1 # Ensure at least 1 dimension if PCA failed badly
         route_latents_rnn_pca_transformed = np.zeros((route_mat_rnn.shape[0], 1))


    # Store PCA'd latents in a dict similar to route_emb_rnn
    route_latents_rnn_dict = {r_id: route_latents_rnn_pca_transformed[i] for i, r_id in enumerate(route_ids_rnn)}

    # KNN on these PCA'd RNN latents (this will be updated later after NCF merge)
    if route_latents_rnn_pca_transformed.shape[0] > 0 and orig_latent_dim > 0:
         # n_neighbors for KNN should be less than number of samples
        knn_n_neighbors = min(30, route_latents_rnn_pca_transformed.shape[0])
        if knn_n_neighbors > 0:
            knn_real_initial = NearestNeighbors(n_neighbors=knn_n_neighbors).fit(route_latents_rnn_pca_transformed)
        else:
            knn_real_initial = None # Cannot fit KNN
            print("Warning: Not enough samples to fit KNN on initial RNN latents.")
    else:
        knn_real_initial = None
        print("Warning: RNN latents are empty or zero-dimensional before NCF merge. KNN not fitted.")

    route_index = {r: i for i, r in enumerate(route_ids_rnn)} # Index based on rnn processed routes

# -----------------------------------------------
# 10. NCF 모델 생성 및 학습 (ENHANCED)
# -----------------------------------------------
all_user_ids = df['userID'].unique().tolist()
all_item_ids = df['itemID'].unique().tolist() # These are the itemIDs used by NCF

user2idx = {u: i for i, u in enumerate(all_user_ids)}
item2idx = {it: i for i, it in enumerate(all_item_ids)} # itemID to NCF's internal item index
idx2item = {i: it for it, i in item2idx.items()} # NCF internal index to itemID


df['user_idx'] = df['userID'].map(user2idx)
df['item_idx'] = df['itemID'].map(item2idx)

user_pos_set = df.groupby('user_idx')['item_idx'].apply(set).to_dict()

users_ncf, items_ncf, labels_ncf = [], [], []
num_negatives = 4

for u_idx, item_indices in user_pos_set.items():
    for i_idx in item_indices:
        users_ncf.append(u_idx)
        items_ncf.append(i_idx)
        labels_ncf.append(1.0)
        for _ in range(num_negatives):
            neg_i = np.random.randint(0, len(all_item_ids))
            while neg_i in item_indices: # Ensure negative is not in positive set for this user
                neg_i = np.random.randint(0, len(all_item_ids))
            users_ncf.append(u_idx)
            items_ncf.append(neg_i)
            labels_ncf.append(0.0)

users_ncf = np.array(users_ncf, dtype=np.int32)
items_ncf = np.array(items_ncf, dtype=np.int32)
labels_ncf = np.array(labels_ncf, dtype=np.float32)


# Helper function for route features for NCF
def get_route_ncf_features(item_id_str, df_main, mv_cols, p_themes, p_subthemes, p_to_coord, th_idx, subth_idx, v_places, th_dim, subth_dim):
    route_row = df_main[df_main['itemID'] == item_id_str]
    if route_row.empty:
        return {
            'avg_theme_vec': np.zeros(th_dim), 'avg_subtheme_vec': np.zeros(subth_dim),
            'poi_count': 0.0, 'total_distance': 0.0, 'geo_centroid': np.zeros(2)
        }

    poi_sequence = []
    # Iterate over movingpath columns for the specific itemID row
    for col in mv_cols:
        if col in route_row.columns:
            val = route_row[col].iloc[0] # Get value from the first (and assumed only) row for this itemID
            if pd.notnull(val) and val in v_places:
                poi_sequence.append(val)
    
    if not poi_sequence:
         return {
            'avg_theme_vec': np.zeros(th_dim), 'avg_subtheme_vec': np.zeros(subth_dim),
            'poi_count': 0.0, 'total_distance': 0.0, 'geo_centroid': np.zeros(2)
        }

    route_theme_vectors = []
    route_subtheme_vectors = []
    for p_name in poi_sequence:
        p_theme_vec = np.zeros(th_dim)
        for theme_name in p_themes.get(p_name, set()):
            if theme_name in th_idx and theme_name not in exclude_themes: # Check exclusion here too
                p_theme_vec[th_idx[theme_name]] = 1 
        route_theme_vectors.append(p_theme_vec)
        
        p_subtheme_vec = np.zeros(subth_dim)
        for subtheme_name in p_subthemes.get(p_name, set()):
            if subtheme_name in subth_idx:
                p_subtheme_vec[subth_idx[subtheme_name]] = 1
        route_subtheme_vectors.append(p_subtheme_vec)

    avg_th_vec = np.mean(route_theme_vectors, axis=0) if route_theme_vectors else np.zeros(th_dim)
    avg_subth_vec = np.mean(route_subtheme_vectors, axis=0) if route_subtheme_vectors else np.zeros(subth_dim)
    
    p_count = float(len(poi_sequence))
    
    r_dist = 0.0
    if p_count > 1:
        for i in range(len(poi_sequence) - 1):
            p1_name, p2_name = poi_sequence[i], poi_sequence[i+1]
            if p1_name in p_to_coord and p2_name in p_to_coord:
                lat1_val, lon1_val = p_to_coord[p1_name]
                lat2_val, lon2_val = p_to_coord[p2_name]
                r_dist += haversine(lat1_val, lon1_val, lat2_val, lon2_val)
                
    coords_list = [p_to_coord[p_name] for p_name in poi_sequence if p_name in p_to_coord]
    g_centroid = np.mean(coords_list, axis=0) if coords_list else np.zeros(2)
    
    return {
        'avg_theme_vec': avg_th_vec, 'avg_subtheme_vec': avg_subth_vec,
        'poi_count': p_count, 'total_distance': float(r_dist), 'geo_centroid': g_centroid
    }

# Prepare additional features for NCF
user_features_for_ncf_list = []
for u_ncf_idx in users_ncf: # users_ncf contains user_idx (0 to num_users-1)
    original_user_id = all_user_ids[u_ncf_idx] # Map NCF index back to original userID
    
    theme_freq_v = get_user_theme_freq_vector(original_user_id)
    subtheme_freq_v = get_user_subtheme_freq_vector(original_user_id)
    geo_center_v = get_user_geo_center(original_user_id)
    if geo_center_v is None: geo_center_v = np.zeros(2)
        
    user_feat_concat = np.concatenate([theme_freq_v, subtheme_freq_v, geo_center_v])
    user_features_for_ncf_list.append(user_feat_concat)

# Precompute all item features
item_features_cache_ncf = {}
for item_id_original in all_item_ids: # Iterate over original itemID strings
    # Call helper with all necessary precomputed maps/indices
    feats_dict = get_route_ncf_features(
        item_id_original, df, moving_cols, 
        place_themes, place_subthemes, place_to_coord, 
        theme_index, subtheme_index, valid_places,
        theme_dim_orig, subtheme_dim_orig
    )
    item_features_cache_ncf[item_id_original] = np.concatenate([
        feats_dict['avg_theme_vec'],
        feats_dict['avg_subtheme_vec'],
        np.array([feats_dict['poi_count']]),
        np.array([feats_dict['total_distance']]),
        feats_dict['geo_centroid']
    ])

item_features_for_ncf_list = []
for i_ncf_idx in items_ncf: # items_ncf contains item_idx (0 to num_items-1)
    original_item_id = idx2item[i_ncf_idx] # Map NCF index back to original itemID string
    item_features_for_ncf_list.append(item_features_cache_ncf[original_item_id])


user_features_input_ncf = np.array(user_features_for_ncf_list, dtype=np.float32)
item_features_input_ncf = np.array(item_features_for_ncf_list, dtype=np.float32)


# Scale numerical parts of item_features_input_ncf (POI count and total_distance)
# poi_count is at index: theme_dim_orig + subtheme_dim_orig
# total_distance is at index: theme_dim_orig + subtheme_dim_orig + 1
idx_poi_count = theme_dim_orig + subtheme_dim_orig
idx_distance = idx_poi_count + 1

if item_features_input_ncf.shape[0] > 0 : # Check if array is not empty
    scaler_item_feat_poi_count = StandardScaler()
    item_features_input_ncf[:, idx_poi_count] = scaler_item_feat_poi_count.fit_transform(
        item_features_input_ncf[:, idx_poi_count].reshape(-1,1)
    ).flatten()

    scaler_item_feat_distance = StandardScaler()
    item_features_input_ncf[:, idx_distance] = scaler_item_feat_distance.fit_transform(
        item_features_input_ncf[:, idx_distance].reshape(-1,1)
    ).flatten()
else:
    print("Warning: item_features_input_ncf is empty. Skipping scaling.")


user_add_feat_dim_ncf = user_features_input_ncf.shape[1] if user_features_input_ncf.ndim > 1 and user_features_input_ncf.shape[0] > 0 else 0
item_add_feat_dim_ncf = item_features_input_ncf.shape[1] if item_features_input_ncf.ndim > 1 and item_features_input_ncf.shape[0] > 0 else 0


def build_ncf_model_enhanced(num_users_ncf, num_items_ncf, 
                             user_additional_dim, item_additional_dim,
                             embedding_dim_ncf=32, mlp_hidden_layers=[64, 32, 16], mlp_dropout_rate=0.2):
    
    user_id_input_layer = Input(shape=(1,), dtype='int32', name='user_id_input')
    item_id_input_layer = Input(shape=(1,), dtype='int32', name='item_id_input')
    
    user_feat_input_layer = Input(shape=(user_additional_dim,), dtype='float32', name='user_features_input')
    item_feat_input_layer = Input(shape=(item_additional_dim,), dtype='float32', name='item_features_input')

    # Embedding layers for IDs
    user_id_embedding_layer = KerasEmbedding(
        input_dim=num_users_ncf, output_dim=embedding_dim_ncf,
        name='user_id_embedding', input_length=1
    )
    item_id_embedding_layer = KerasEmbedding(
        input_dim=num_items_ncf, output_dim=embedding_dim_ncf,
        name='item_id_embedding', input_length=1
    )

    user_id_vector = Flatten()(user_id_embedding_layer(user_id_input_layer))
    item_id_vector = Flatten()(item_id_embedding_layer(item_id_input_layer))

    # Concatenate ID embeddings with their respective features
    user_full_vector = Concatenate(name='user_combined_vector')([user_id_vector, user_feat_input_layer])
    item_full_vector = Concatenate(name='item_combined_vector')([item_id_vector, item_feat_input_layer])
    
    # MLP tower
    # Option 1: GMF-style interaction on ID embeddings, then concat with features for MLP (more standard NCF)
    # gmf_interaction = Multiply()([user_id_vector, item_id_vector])
    # mlp_input_concat = Concatenate()([gmf_interaction, user_feat_input_layer, item_feat_input_layer]) # Example

    # Option 2: Interaction over full combined vectors (as per current interpretation of request)
    # Or, simply concatenate all and feed to MLP
    mlp_input_concat = Concatenate(name='interaction_concat_for_mlp')([user_full_vector, item_full_vector])
    
    mlp_layer = mlp_input_concat
    for i, num_units in enumerate(mlp_hidden_layers):
        mlp_layer = Dense(num_units, activation='relu', name=f'mlp_dense_{i}')(mlp_layer)
        mlp_layer = Dropout(mlp_dropout_rate, name=f'mlp_dropout_{i}')(mlp_layer)

    output_prediction = Dense(1, activation='sigmoid', name='prediction')(mlp_layer)

    ncf_enhanced_model = Model(
        inputs=[user_id_input_layer, item_id_input_layer, user_feat_input_layer, item_feat_input_layer], 
        outputs=output_prediction
    )
    
    ncf_enhanced_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return ncf_enhanced_model

num_users_for_ncf = len(all_user_ids)
num_items_for_ncf = len(all_item_ids) # Number of unique itemIDs
embedding_dim_for_ncf = 32 # ID embedding dimension

# Check if dimensions are valid before building model
if user_add_feat_dim_ncf <= 0 or item_add_feat_dim_ncf <= 0:
    print("Error: Additional feature dimensions for NCF are zero or invalid. Cannot build enhanced NCF model.")
    # Fallback or exit
    ncf_model = None 
    # Ensure these fallbacks use the correctly defined variable names too
    user_emb_matrix_ncf = np.zeros((num_users_for_ncf if 'num_users_for_ncf' in locals() else 0, 
                                    embedding_dim_for_ncf if 'embedding_dim_for_ncf' in locals() else 32))
    item_emb_matrix_ncf = np.zeros((num_items_for_ncf if 'num_items_for_ncf' in locals() else 0, 
                                    embedding_dim_for_ncf if 'embedding_dim_for_ncf' in locals() else 32))
else:
    ncf_model = build_ncf_model_enhanced(
        num_users_for_ncf, num_items_for_ncf, # CORRECTED: num_users_for_ncf
        user_add_feat_dim_ncf, item_add_feat_dim_ncf,
        embedding_dim_ncf=embedding_dim_for_ncf,
        mlp_hidden_layers=[128, 64, 32], 
        mlp_dropout_rate=0.3
    )
    if ncf_model: # Check if model was successfully created
        ncf_model.summary()

        # Split data for NCF training (if using validation split)
        # This should be done on the indices and then map to features
        u_train, u_val, i_train, i_val, uf_train, uf_val, if_train, if_val, l_train, l_val = train_test_split(
            users_ncf, items_ncf, 
            user_features_input_ncf, item_features_input_ncf, 
            labels_ncf, 
            test_size=0.1, random_state=42
        )
        
        ncf_model.fit(
            [u_train, i_train, uf_train, if_train], l_train,
            validation_data=([u_val, i_val, uf_val, if_val], l_val),
            batch_size=256, 
            epochs=10, 
            verbose=1
        )

        user_emb_matrix_ncf = ncf_model.get_layer('user_id_embedding').get_weights()[0]
        item_emb_matrix_ncf = ncf_model.get_layer('item_id_embedding').get_weights()[0]
    else: # Fallback if model building failed for other reasons (e.g. inside build_ncf_model_enhanced)
        print("NCF model could not be built. Using zero embeddings as fallback.")
        user_emb_matrix_ncf = np.zeros((num_users_for_ncf if 'num_users_for_ncf' in locals() else 0, 
                                    embedding_dim_for_ncf if 'embedding_dim_for_ncf' in locals() else 32))
        item_emb_matrix_ncf = np.zeros((num_items_for_ncf if 'num_items_for_ncf' in locals() else 0, 
                                    embedding_dim_for_ncf if 'embedding_dim_for_ncf' in locals() else 32))


# Map NCF embeddings (which are based on NCF's internal 0..N-1 indices) back to original userIDs and itemIDs
user_emb_ncf = {
    all_user_ids[i]: user_emb_matrix_ncf[i]
    for i in range(num_users_for_ncf)
    if i < user_emb_matrix_ncf.shape[0] # Boundary check
}
item_emb_ncf = {
    idx2item[j]: item_emb_matrix_ncf[j] # Use idx2item to map NCF item index to original itemID string
    for j in range(num_items_for_ncf)
    if j < item_emb_matrix_ncf.shape[0] # Boundary check
}

# -----------------------------------------------
# 11. User embedding merging (MetaPath + NCF)
# -----------------------------------------------
# user_dim_meta from Metapath2Vec, embedding_dim_for_ncf from NCF
ncf_emb_dim = embedding_dim_for_ncf # Dimension of NCF ID embeddings
user_final_dim = user_dim_meta + ncf_emb_dim

combined_user_emb = {}
for u_id_orig in all_user_ids: # Iterate using original user IDs
    meta_vec_u = user_emb_meta.get(u_id_orig, np.zeros(user_dim_meta))
    ncf_vec_u  = user_emb_ncf.get(u_id_orig,  np.zeros(ncf_emb_dim))
    combined_user_emb[u_id_orig] = np.concatenate([meta_vec_u, ncf_vec_u])

# -----------------------------------------------
# 12. Route latent merging (RNN → PCA + NCF)
# -----------------------------------------------
# route_latents_rnn_dict: Original itemID string -> PCA'd RNN embedding
# item_emb_ncf: Original itemID string -> NCF ID embedding

route_latents_merged_dict = {}
# Ensure orig_latent_dim (from RNN PCA) and ncf_emb_dim (from NCF) are defined
if orig_latent_dim == 0: # Handle case where RNN pipeline might have failed to produce valid dimensions
    print("Warning: orig_latent_dim (RNN PCA output) is 0. Merged route latents might be based only on NCF.")

for item_id_str in all_item_ids: # Iterate through all unique itemIDs
    # Get RNN latent (PCA'd)
    # It's possible route_latents_rnn_dict doesn't have all item_ids if RNN processing had issues.
    rnn_lat = route_latents_rnn_dict.get(item_id_str, np.zeros(orig_latent_dim if orig_latent_dim > 0 else 0))
    
    # Get NCF latent
    ncf_lat = item_emb_ncf.get(item_id_str, np.zeros(ncf_emb_dim))
    
    # Ensure rnn_lat has a shape even if it was zeros(0)
    if rnn_lat.ndim == 1 and rnn_lat.shape[0] == 0 and orig_latent_dim > 0:
        rnn_lat = np.zeros(orig_latent_dim) # Correct empty array to zero vector of expected dim
    elif rnn_lat.ndim == 0 and orig_latent_dim > 0: # If it was a scalar zero or something
        rnn_lat = np.zeros(orig_latent_dim)


    merged_lat_vec = np.concatenate([rnn_lat, ncf_lat])
    route_latents_merged_dict[item_id_str] = merged_lat_vec

# This latent_dim is for the CGAN's latent space (output of Generator)
latent_dim_cgan = (orig_latent_dim if orig_latent_dim > 0 else 0) + ncf_emb_dim
if latent_dim_cgan == 0:
    print("CRITICAL WARNING: CGAN latent_dim is 0. This will cause errors. Check RNN and NCF embedding pipelines.")
    latent_dim_cgan = 1 # Minimal dimension to prevent immediate crash, but results will be meaningless

# Create the matrix for KNN using the merged latents
# The order of routes in route_latents (for KNN) should be consistent. Use all_item_ids.
route_latents_for_knn = np.vstack([route_latents_merged_dict[r_id] for r_id in all_item_ids if r_id in route_latents_merged_dict])

if route_latents_for_knn.shape[0] > 0 and latent_dim_cgan > 0:
    knn_n_neighbors_final = min(30, route_latents_for_knn.shape[0])
    if knn_n_neighbors_final > 0:
        knn_merged_routes = NearestNeighbors(n_neighbors=knn_n_neighbors_final).fit(route_latents_for_knn)
    else:
        knn_merged_routes = None
        print("Warning: Not enough samples to fit KNN on merged route latents.")
else:
    knn_merged_routes = None # Cannot fit KNN if no data or zero dimension
    print("Warning: Merged route latents are empty or zero-dimensional. KNN for routes not fitted.")

# Update route_index to be based on all_item_ids for consistency with route_latents_for_knn
route_index_final = {r_id: i for i, r_id in enumerate(all_item_ids) if r_id in route_latents_merged_dict}


# ===============================================================
# 13. POI Embedding(place_emb) generation (re-design)
# → Metapath2Vec(64dim) and NCF-based context(32dim) to generate 96dim embedding
# ===============================================================
print("\n[INFO] Section 13: Re-designing POI embeddings (Metapath2Vec + NCF Context)...")
place_to_route_item_ids = defaultdict(list)
for _, row in df.iterrows():
    item_id_of_row = row['itemID']
    route_pois_in_row = [row[mc] for mc in moving_cols if pd.notnull(row[mc]) and row[mc] in valid_places]
    for p_in_route in route_pois_in_row:
        place_to_route_item_ids[p_in_route].append(item_id_of_row)

place_emb = {}
if 'w2v_model' in locals() and 'item_emb_ncf' in locals() and w2v_model is not None and item_emb_ncf:
    meta_dim = w2v_model.vector_size
    ncf_dim = next(iter(item_emb_ncf.values())).shape[0]
    for poi_name in valid_places:
        poi_str = str(poi_name)
        meta_vec = w2v_model.wv[poi_str] if poi_str in w2v_model.wv else np.zeros(meta_dim)
        routes_containing_poi = place_to_route_item_ids.get(poi_name, [])
        ncf_vectors = [item_emb_ncf[r_id] for r_id in routes_containing_poi if r_id in item_emb_ncf]
        context_vec = np.mean(ncf_vectors, axis=0) if ncf_vectors else np.zeros(ncf_dim)
        place_emb[poi_name] = np.concatenate([meta_vec, context_vec])
    place_emb_dim = meta_dim + ncf_dim
    print(f"  ▶ Successfully created {place_emb_dim}-dim hybrid embeddings for {len(place_emb)} POIs.")
else:
    print("  ▶ CRITICAL WARNING: Metapath2Vec or NCF model not found. Using zero vectors for POI embeddings.")
    place_emb = {p: np.zeros(latent_dim_cgan) for p in valid_places}
    place_emb_dim = latent_dim_cgan if 'latent_dim_cgan' in globals() else 1

# -----------------------------------------------
# 13-5. Similar Users(NCF) 및 preference prediction function
# -----------------------------------------------
# Ensure pref_cols are valid and exist in df
print("\n[INFO] Section 13.5: Pre-building Similar User Model...")

if not valid_pref_cols:
    print("  ▶ Warning: No valid preference columns (pref_*) found. User similarity model cannot be built.")
    sim_user_model = None
    user_ids_pref_knn = []
    user_pref_df_cleaned = pd.DataFrame()
else:
    user_pref_df = df.groupby('userID')[valid_pref_cols].mean()
    user_pref_df_cleaned = user_pref_df.fillna(0).replace([np.inf, -np.inf], 0)
    
    n_neighbors_sim_user = min(10, len(user_pref_df_cleaned))
    if n_neighbors_sim_user > 0:
        sim_user_model = NearestNeighbors(metric='cosine').fit(user_pref_df_cleaned.values)
        print(f"  ▶ Successfully built Similar User Model (k={n_neighbors_sim_user}).")
    else:
        sim_user_model = None
        print("  ▶ Warning: Not enough users to fit similar user model (KNN).")
    user_ids_pref_knn = user_pref_df_cleaned.index.tolist()

# POI Similarity Model generation
print("  ▶ Building POI-to-POI similarity model (KNN)...")
poi_list_for_knn = list(place_emb.keys())
poi_matrix_for_knn = np.array([place_emb[p] for p in poi_list_for_knn])
if poi_matrix_for_knn.shape[0] > 0:
    n_neighbors_poi = min(50, len(poi_list_for_knn))
    poi_sim_model = NearestNeighbors(n_neighbors=n_neighbors_poi, metric='cosine').fit(poi_matrix_for_knn)
    print(f"  ▶ Successfully built POI similarity model (k={n_neighbors_poi}).")
else:
    poi_sim_model = None
    print("  ▶ Warning: Not enough POI embeddings to build similarity model.")

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

def compute_fid(real_feats, fake_feats):
    mu_r, mu_f = real_feats.mean(axis=0), fake_feats.mean(axis=0)
    sigma_r = np.cov(real_feats, rowvar=False)
    sigma_f = np.cov(fake_feats, rowvar=False)
    return calculate_frechet_distance(mu_r, sigma_r, mu_f, sigma_f)

def compute_inception_score(feats, n_splits=10, eps=1e-16):
    gmm = GaussianMixture(n_components=n_splits, covariance_type='full', random_state=0)
    gmm.fit(feats)
    preds = gmm.predict_proba(feats)
    py = preds.mean(axis=0)
    kl = preds * (np.log(preds + eps) - np.log(py[None, :] + eps))
    return np.exp(np.mean(np.sum(kl, axis=1)))

def compute_fjd(real_feats, real_conds, fake_feats, fake_conds):
    """
    FJD: joint distribution Fréchet distance for (x, c)
    real_feats/fake_feats: (N, D_feat), real_conds/fake_conds: (N, D_cond)
    """
    real_joint = np.hstack([real_feats, real_conds])
    fake_joint = np.hstack([fake_feats, fake_conds])
    mu_r, mu_f = real_joint.mean(axis=0), fake_joint.mean(axis=0)
    sigma_r = np.cov(real_joint, rowvar=False)
    sigma_f = np.cov(fake_joint, rowvar=False)
    return calculate_frechet_distance(mu_r, sigma_r, mu_f, sigma_f)

def rbf_kernel(X, Y, gamma):
    """RBF Kernel: k(x,y)=exp(-gamma * ||x-y||^2)"""
    X_norm = np.sum(X**2, axis=1)[:, None]
    Y_norm = np.sum(Y**2, axis=1)[None, :]
    sq_dists = X_norm + Y_norm - 2 * X.dot(Y.T)
    return np.exp(-gamma * sq_dists)

def compute_mmd_rkhs(X, Y, gamma=None, unbiased=True):
    """
    RKHS 기반 MMD calculation (RBF kernel, unbiased estimation).
    - X: (n, d) real latent vectors
    - Y: (m, d) generated latent vectors
    """
    n, m = X.shape[0], Y.shape[0]
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    Kxx = rbf_kernel(X, X, gamma)
    Kyy = rbf_kernel(Y, Y, gamma)
    Kxy = rbf_kernel(X, Y, gamma)

    if unbiased:
        np.fill_diagonal(Kxx, 0)
        np.fill_diagonal(Kyy, 0)
        mmd2 = (Kxx.sum()   / (n*(n-1))
              + Kyy.sum()   / (m*(m-1))
              - 2 * Kxy.sum() / (n*m))
    else:
        mmd2 = Kxx.mean() + Kyy.mean() - 2*Kxy.mean()

    return np.sqrt(max(mmd2, 0.0))

def compute_wasserstein_distance(real_feats, fake_feats):
    """
    Wasserstein Distance calculation (for multi-dimensional features)
    real_feats, fake_feats: (N, D) shape feature matrix
    """
    if real_feats.shape[1] != fake_feats.shape[1]:
        raise ValueError("Feature dimensions must match")
    
    wd_per_dim = []
    for dim in range(real_feats.shape[1]):
        real_dim = real_feats[:, dim]
        fake_dim = fake_feats[:, dim]
        
        real_dim = real_dim[~np.isnan(real_dim)]
        fake_dim = fake_dim[~np.isnan(fake_dim)]
        
        if len(real_dim) > 0 and len(fake_dim) > 0:
            wd = wasserstein_distance(real_dim, fake_dim)
            wd_per_dim.append(wd)
    
    return np.mean(wd_per_dim) if wd_per_dim else 0.0

def get_user_ground_truth_pois(user_id, df, moving_cols):
    """
    Extracts the user's entire visit history as ground truth.
    This is used to evaluate how related the recommendation results are to the user's overall preferences.
    """
    user_rows = df[df['userID'] == user_id]
    ground_truth_pois = set()
    for _, row in user_rows.iterrows():
        for col in moving_cols:
            poi = row[col]
            if pd.notnull(poi):
                ground_truth_pois.add(normalize_poi_name(poi))
    return ground_truth_pois   

def evaluate_performance_by_k_with_fid(k_range, num_gen_routes_per_k=100):
    """
    Based on all enhanced dataset, it evluates the performance of FID with POI number.
    - FID (Fréchet Inception Distance): actual data and generated data distribution distance.
      GAN's standard evaluation metric. Lower score, good performance.
    """
    fid_scores = []

    all_real_routes = []
    for _, row in df.iterrows():
        route = [row[col] for col in moving_cols if pd.notnull(row[col]) and row[col] in place_emb]
        if route:
            all_real_routes.append(route)
    print(f"▶ Found {len(all_real_routes)} real routes in the dataset.")

    candidate_users = df['userID'].unique()
    if len(candidate_users) == 0:
        print("Error: No users found for evaluation.")
        return k_range, [np.nan] * len(k_range)

    for k in k_range:
        print(f"\n--- POI Number(k) = {k} evaluating.. ---")
        real_routes_filtered = [r for r in all_real_routes if len(r) == k]
        
        if len(real_routes_filtered) < 10:
            print(f"  ▶ Warning: Available real routes for k={k} is less than ({len(real_routes_filtered)}). skipping evaluation.")
            fid_scores.append(np.nan)
            continue
            
        real_feats = np.array([np.mean([place_emb[p] for p in route], axis=0) for route in real_routes_filtered])
        print(f"  ▶ Converted k={k} routes to real features: {len(real_feats)} routes.")

        fake_seqs = []
        generation_attempts = 0
        while len(fake_seqs) < num_gen_routes_per_k and generation_attempts < num_gen_routes_per_k * 5:
            random_user_id = random.choice(candidate_users)
            recs = recommend_routes_enhanced(random_user_id, top_k_pois_per_route=k, num_gen_routes=1)
            generation_attempts += 1
            if recs and recs.get('recommended_routes'):
                for r in recs['recommended_routes']:
                    if len(r['sequence']) == k:
                        fake_seqs.append(r['sequence'])
        
        if len(fake_seqs) < 10:
            print(f"  ▶ Warning: Available generated routes for k={k} is less than ({len(fake_seqs)}). skipping evaluation.")
            fid_scores.append(np.nan)
            continue
            
        fake_feats_list = [np.mean([place_emb[p] for p in r_seq if p in place_emb], axis=0) for r_seq in fake_seqs if any(p in place_emb for p in r_seq)]
        
        if len(fake_feats_list) < 10:
            print(f"  ▶ Warning: Available generated routes for k={k} is less than ({len(fake_feats_list)}). skipping evaluation.")
            fid_scores.append(np.nan)
            continue

        fake_feats = np.array(fake_feats_list)
        print(f"  ▶ k={k} refers to generated routes {len(fake_feats)}generate piece and converted.")
        
        min_samples = min(len(real_feats), len(fake_feats))
        real_feats_sample = real_feats[np.random.choice(len(real_feats), min_samples, replace=False)]
        fake_feats_sample = fake_feats[np.random.choice(len(fake_feats), min_samples, replace=False)]
        
        fid_val = compute_fid(real_feats_sample, fake_feats_sample)
        fid_scores.append(fid_val)
        print(f"  ▶ Final performance (k={k}): FID = {fid_val:.4f} (lower the better)")

    return k_range, fid_scores

# -----------------------------------------------
# 13.5. Enhanced Condition Vector generated function (NEW)
# -----------------------------------------------

def create_enhanced_condition_vector(user_id, df, moving_cols, confidence_scores):
    user_combined_v = combined_user_emb.get(user_id, np.zeros(user_final_dim))
    
    multi_signal_features = compute_multi_signal_features(user_id, df, moving_cols, confidence_scores)
    
    user_confidence = confidence_scores.get(user_id, {})
    if user_confidence:
        weighted_freq = sum(
            freq_info.get('frequency_conf', 0.5) * 
            df[df['userID'] == user_id]['totalFreq'].mean()
            for freq_info in user_confidence.values()
        ) / len(user_confidence)
    else:
        weighted_freq = df[df['userID'] == user_id]['totalFreq'].mean()
    
    if pd.isna(weighted_freq):
        weighted_freq = df['totalFreq'].mean()
    
    enhanced_condition_vector = np.concatenate([
        user_combined_v,
        np.array([weighted_freq]),
        multi_signal_features
    ]).astype(np.float32)
    
    return enhanced_condition_vector

def find_similar_users_by_pref(target_user_id, sim_model, user_ids_knn_list, top_k=5):
    if target_user_id not in user_ids_knn_list or sim_model is None:
        return []
    
    target_user_idx = user_ids_knn_list.index(target_user_id)
    target_vector = user_pref_df_cleaned.iloc[target_user_idx].values.reshape(1, -1)
    
    distances, indices = sim_model.kneighbors(target_vector, n_neighbors=top_k + 1)
    
    similar_users = []
    for i in range(1, len(indices[0])):
        sim_user_idx = indices[0][i]
        sim_user_id = user_ids_knn_list[sim_user_idx]
        similarity = 1 - distances[0][i]
        similar_users.append((sim_user_id, similarity))
        
    return similar_users

def create_collaborative_condition_vector(user_id, sim_model, user_ids_list, sparsity_threshold=5):
    original_cond_vec = create_enhanced_condition_vector(user_id, df, moving_cols, all_confidence_scores)
    user_interaction_count = len(df[df['userID'] == user_id])
    if user_interaction_count >= sparsity_threshold:
        return original_cond_vec

    print(f"    [Info] User {user_id} is sparse. Augmenting condition with similar users.")
    similar_users = find_similar_users_by_pref(user_id, sim_model, user_ids_list, top_k=5)
    if not similar_users:
        return original_cond_vec

    similar_user_vectors, total_similarity = [], 0
    for sim_user_id, similarity in similar_users:
        sim_user_vec = create_enhanced_condition_vector(sim_user_id, df, moving_cols, all_confidence_scores)
        similar_user_vectors.append(sim_user_vec * similarity)
        total_similarity += similarity
    if not similar_user_vectors or total_similarity == 0:
        return original_cond_vec
    aggregated_vector = np.sum(similar_user_vectors, axis=0) / total_similarity
    alpha = user_interaction_count / sparsity_threshold
    final_collaborative_vec = (alpha * original_cond_vec) + ((1 - alpha) * aggregated_vector)

    return final_collaborative_vec.astype(np.float32) 

print("  ▶ Building POI-to-POI similarity model (KNN)...")
poi_list_for_knn = list(place_emb.keys())
poi_matrix_for_knn = np.array([place_emb[p] for p in poi_list_for_knn])

if poi_matrix_for_knn.shape[0] > 0:
    n_neighbors_poi = min(50, len(poi_list_for_knn)) # 각 POI 당 최대 50개의 유사 POI 탐색
    poi_sim_model = NearestNeighbors(n_neighbors=n_neighbors_poi, metric='cosine').fit(poi_matrix_for_knn)
    print(f"  ▶ Successfully built POI similarity model (k={n_neighbors_poi}).")
else:
    poi_sim_model = None
    print("  ▶ Warning: Not enough POI embeddings to build similarity model.")    

# -----------------------------------------------
# 14. X_lat / X_cond Prepare & Standardization (for CGAN)
# -----------------------------------------------
print("Computing confidence scores for all users...")
all_confidence_scores = {}
for user_id in df['userID'].unique():
    all_confidence_scores[user_id] = calculate_interaction_confidence(
        user_id, df, moving_cols, poiFreq_cols
    )

print("Building collaborative condition vectors for CGAN...")
X_lat_list, X_cond_list = [], []

for _, row in df.iterrows():
    u_id_orig, item_id_orig_ = row['userID'], row['itemID'] # Original IDs

    if u_id_orig not in combined_user_emb or item_id_orig_ not in route_latents_merged_dict:
        continue

    lat_vec_item = route_latents_merged_dict[item_id_orig_]
    X_lat_list.append(lat_vec_item)

    cond_vec_user = create_collaborative_condition_vector(
        u_id_orig, sim_user_model, user_ids_pref_knn, sparsity_threshold=5
    )
    X_cond_list.append(cond_vec_user)


if not X_lat_list or not X_cond_list:
    print("CRITICAL: X_lat_list or X_cond_list is empty. CGAN cannot be trained.")
else:
    X_lat = np.vstack(X_lat_list).astype(np.float32)
    X_cond = np.vstack(X_cond_list).astype(np.float32)

    print(f"Built latent vectors X_lat with shape: {X_lat.shape}")
    print(f"Built condition vectors X_cond with shape: {X_cond.shape}")
    
    X_lat_train, X_lat_test, X_cond_train, X_cond_test = train_test_split(
        X_lat, X_cond, test_size=0.2, random_state=42
    )

    scaler_lat  = MinMaxScaler(feature_range=(-1, 1)).fit(X_lat_train)
    scaler_cond = StandardScaler().fit(X_cond_train)

    n_quantiles_for_transform = min(1000, X_lat_train.shape[0] // 10)
    quantile_transformer = QuantileTransformer(output_distribution='normal', 
                                            n_quantiles=n_quantiles_for_transform,
                                            random_state=42)
    quantile_transformer.fit(X_lat_train[:, [0]])

    print("  ▶ QuantileTransformer for dimension 0 has been fitted.")

    X_lat_train_scaled = scaler_lat.transform(X_lat_train) if X_lat_train.shape[0] > 0 else X_lat_train
    X_lat_test_scaled  = scaler_lat.transform(X_lat_test) if X_lat_test.shape[0] > 0 else X_lat_test

    if X_lat_train.shape[0] > 0:
        X_lat_train_scaled[:, 0] = quantile_transformer.transform(X_lat_train[:, [0]]).flatten()
    if X_lat_test.shape[0] > 0:
        X_lat_test_scaled[:, 0] = quantile_transformer.transform(X_lat_test[:, [0]]).flatten()

    X_cond_train_scaled= scaler_cond.transform(X_cond_train) if X_cond_train.shape[0] > 0 else X_cond_train
    X_cond_test_scaled = scaler_cond.transform(X_cond_test) if X_cond_test.shape[0] > 0 else X_cond_test


    cond_dim  = X_cond_train_scaled.shape[1]
    noise_dim = 128 

    X_lat_train_cgan_input = X_lat_train_scaled


if not X_lat_list or not X_cond_list:
    print("CRITICAL: X_lat_list or X_cond_list is empty. CGAN cannot be trained.")
else:
    X_lat = np.vstack(X_lat_list).astype(np.float32)
    X_cond = np.vstack(X_cond_list).astype(np.float32)

    print(f"Enhanced condition vectors shape: {X_cond.shape}")
    
    X_lat_train, X_lat_test, X_cond_train, X_cond_test = train_test_split(
        X_lat, X_cond, test_size=0.2, random_state=42
    )

    scaler_lat  = MinMaxScaler(feature_range=(-1, 1)).fit(X_lat_train)
    scaler_cond = StandardScaler().fit(X_cond_train)

    X_lat_train_scaled = scaler_lat.transform(X_lat_train) if X_lat_train.shape[0] > 0 else X_lat_train
    X_lat_test_scaled  = scaler_lat.transform(X_lat_test) if X_lat_test.shape[0] > 0 else X_lat_test
    X_cond_train_scaled= scaler_cond.transform(X_cond_train) if X_cond_train.shape[0] > 0 else X_cond_train
    X_cond_test_scaled = scaler_cond.transform(X_cond_test) if X_cond_test.shape[0] > 0 else X_cond_test


    cond_dim  = X_cond_train_scaled.shape[1]
    noise_dim = 128

    X_lat_train_cgan_input = X_lat_train_scaled

class SpectralNormalization(Wrapper):
    def __init__(self, layer, power_iterations=1, **kwargs):
        super().__init__(layer, **kwargs)
        self.power_iterations = power_iterations

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.initializers.RandomNormal(),
            trainable=False,
            name='u_vector',
            dtype=self.w.dtype,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u_hat = self.u

        for _ in range(self.power_iterations):
            # v_hat = l2_normalize(u_hat @ W.T)
            v_hat = tf.math.l2_normalize(tf.matmul(u_hat, w_reshaped, transpose_b=True))
            # u_hat = l2_normalize(v_hat @ W)
            u_hat = tf.math.l2_normalize(tf.matmul(v_hat, w_reshaped))

        if training:
            self.u.assign(u_hat)

        # Spectral Nom(sigma) calculate: sigma = v.T * W * u
        # shapes: v_hat(1, in_dim), w_reshaped(in_dim, out_dim), u_hat(1, out_dim)
        sigma = tf.matmul(v_hat, tf.matmul(w_reshaped, u_hat, transpose_b=True))
        sigma = sigma[0, 0]

        normalized_kernel = self.w / sigma
        outputs = tf.matmul(inputs, normalized_kernel)
        
        if self.layer.use_bias:
            outputs = tf.nn.bias_add(outputs, self.layer.bias)
        if self.layer.activation is not None:
            outputs = self.layer.activation(outputs)
            
        return outputs

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

# -----------------------------------------------
# 15. CGAN Generator / Discriminator
# -----------------------------------------------
def build_generator_v2(noise_d, cond_d, out_d):
    z_input = Input(shape=(noise_d,), name="gen_noise_input")
    g_z = Dense(256)(z_input)
    g_z = BatchNormalization()(g_z)
    g_z = LeakyReLU(alpha=0.2)(g_z)

    c_input = Input(shape=(cond_d,), name="gen_condition_input")
    g_c = Dense(256)(c_input)
    g_c = BatchNormalization()(g_c)
    g_c = LeakyReLU(alpha=0.2)(g_c)

    concat = Concatenate()([g_z, g_c])
    
    x = Dense(512)(concat)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    out_layer = Dense(out_d, activation='tanh')(x)
    return Model([z_input, c_input], out_layer, name="Generator")

def build_condition_encoder(lat_d, cond_d):
    latent_input = Input(shape=(lat_d,), name="encoder_latent_input")
    
    e = Dense(256)(latent_input)
    e = LeakyReLU(alpha=0.2)(e)
    
    e = Dense(512)(e)
    e = LeakyReLU(alpha=0.2)(e)
    
    cond_output = Dense(cond_d, activation='linear')(e)
    
    return Model(latent_input, cond_output, name="ConditionEncoder")    


def build_disc_spectral(lat_d, cond_d):
    """
    Apply Spectral Normalization to improve discriminator training stability.
    Wrap Dense layers with SpectralNormalization.
    """
    latent_input = Input((lat_d,), name="disc_latent_input")
    c_input = Input((cond_d,), name="disc_condition_input")

    noisy_latent = GaussianNoise(0.05)(latent_input)
    
    x = Concatenate()([noisy_latent, c_input])

    x = SpectralNormalization(Dense(512))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)

    x = SpectralNormalization(Dense(256))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)

    validity_output = Dense(1, activation='sigmoid')(x)
    
    return Model([latent_input, c_input], validity_output, name="Discriminator")    

if latent_dim_cgan <= 0 or cond_dim <= 0:
    print("CRITICAL ERROR: CGAN dimensions are invalid.")
    G_lat, D_lat, E_cond, CGAN_model = None, None, None, None
else:

    g_optimizer = Adam(0.0003, beta_1=0.5)
    d_optimizer = Adam(0.0001, beta_1=0.5)
    e_optimizer = Adam(0.0002, beta_1=0.5)

    D_lat = build_disc_spectral(latent_dim_cgan, cond_dim)
    D_lat.compile(optimizer=d_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    G_lat = build_generator_v2(noise_dim, cond_dim, latent_dim_cgan)
    E_cond = build_condition_encoder(latent_dim_cgan, cond_dim)
    
    D_lat.trainable = False
    
    z_gan_input = Input((noise_dim,), name="cgan_noise_input")
    c_gan_input = Input((cond_dim,), name="cgan_condition_input")
    
    fake_latent_output = G_lat([z_gan_input, c_gan_input])
    
    validity_from_d = D_lat([fake_latent_output, c_gan_input])
    reconstructed_cond = E_cond(fake_latent_output) # generated result to reconstruct condition
    
    CGAN_model = Model(
        [z_gan_input, c_gan_input], 
        [validity_from_d, reconstructed_cond], 
        name="CGAN_with_Encoder"
    )

    G_lat.summary()
    D_lat.summary()
    E_cond.summary()
    CGAN_model.summary()

# -----------------------------------------------
# 16. CGAN training loop
# -----------------------------------------------
batch_size_cgan = 128
epochs_cgan = 5000
real_label_smooth = 0.9
lambda_cond = 0.1

dim_wise_errors = tf.Variable(tf.ones(latent_dim_cgan), trainable=False, dtype=tf.float32)
loss_weights = tf.Variable(tf.ones(latent_dim_cgan), trainable=False, dtype=tf.float32)
ema_decay = 0.99

d_losses_hist, g_losses_hist, c_losses_hist, d_accs_hist = [], [], [], []

bce_loss = tf.keras.losses.BinaryCrossentropy()

if G_lat and D_lat and E_cond and CGAN_model \
   and X_lat_train_cgan_input.shape[0] > 0 \
   and X_cond_train_scaled.shape[0] > 0:

    print("\n--- Starting CGAN Training with Spectral Norm & Progressive Weighted Loss ---")
    for epoch_num in range(1, epochs_cgan + 1):
        
        # ==========================================================
        # 1. Discriminator Training
        # ==========================================================
        D_lat.trainable = True
        idx_batch = np.random.randint(0, X_lat_train_cgan_input.shape[0], batch_size_cgan)
        real_lat_batch  = tf.convert_to_tensor(X_lat_train_cgan_input[idx_batch], dtype=tf.float32)
        real_cond_batch = tf.convert_to_tensor(X_cond_train_scaled[idx_batch], dtype=tf.float32)
        z_noise_d = tf.random.normal((batch_size_cgan, noise_dim), dtype=tf.float32)
        fake_lat_batch_d = G_lat([z_noise_d, real_cond_batch], training=False)
        
        d_loss_real, d_acc_real = D_lat.train_on_batch([real_lat_batch, real_cond_batch], tf.fill((batch_size_cgan, 1), real_label_smooth))
        d_loss_fake, d_acc_fake = D_lat.train_on_batch([fake_lat_batch_d, real_cond_batch], tf.zeros((batch_size_cgan, 1)))
        d_loss_epoch = 0.5 * (d_loss_real + d_loss_fake)
        d_acc_epoch = 0.5 * (d_acc_real + d_acc_fake)

        # ==========================================================
        # 2. Generator & Encoder Training
        # ==========================================================
        D_lat.trainable = False
        idx_g_batch = np.random.randint(0, X_cond_train_scaled.shape[0], batch_size_cgan)
        cond_batch_g = tf.convert_to_tensor(X_cond_train_scaled[idx_g_batch], dtype=tf.float32)
        real_lat_g_batch = tf.convert_to_tensor(X_lat_train_cgan_input[idx_g_batch], dtype=tf.float32)
        z_noise_g = tf.random.normal((batch_size_cgan, noise_dim), dtype=tf.float32)

        with tf.GradientTape() as g_tape:
            fake_latents = G_lat([z_noise_g, cond_batch_g], training=True)
            fake_validity = D_lat([fake_latents, cond_batch_g], training=False)
            reconstructed_conds = E_cond(fake_latents, training=True)

            adversarial_loss = bce_loss(tf.ones_like(fake_validity), fake_validity)

            reconstruction_error = tf.square(cond_batch_g - reconstructed_conds)
            condition_loss = tf.reduce_mean(reconstruction_error)

            error_per_dim = tf.abs(fake_latents - real_lat_g_batch)
            progressive_weighted_loss = tf.reduce_mean(loss_weights * tf.square(error_per_dim))

            lambda_progressive = 0.1
            total_g_loss = adversarial_loss + lambda_cond * condition_loss + lambda_progressive * progressive_weighted_loss

        g_trainable_vars = G_lat.trainable_variables + E_cond.trainable_variables
        gradients = g_tape.gradient(total_g_loss, g_trainable_vars)
        g_optimizer.apply_gradients(zip(gradients, g_trainable_vars))

        current_batch_error = tf.reduce_mean(error_per_dim, axis=0)
        dim_wise_errors.assign(ema_decay * dim_wise_errors + (1 - ema_decay) * current_batch_error)
        new_weights = dim_wise_errors / (tf.reduce_mean(dim_wise_errors) + 1e-8)
        loss_weights.assign(new_weights)

        # -----------------
        # 3. Record & Print
        # -----------------
        d_losses_hist.append(d_loss_epoch)
        g_losses_hist.append(adversarial_loss.numpy())
        c_losses_hist.append(condition_loss.numpy())
        d_accs_hist.append(d_acc_epoch)
        
        if epoch_num % 500 == 0:
            print(f"Epoch {epoch_num}/{epochs_cgan} | "
                  f"D_loss: {d_loss_epoch:.4f} | "
                  f"D_acc: {d_acc_epoch*100:.2f}% | "
                  f"G_loss_adv: {adversarial_loss:.4f} | "
                  f"G_loss_cond (weighted): {condition_loss:.4f}") 

    # --- fake_latent generation for visualization ---
    noise_vis = np.random.normal(0, 1, (X_lat_train_cgan_input.shape[0], noise_dim))
    fake_latent_scaled = G_lat.predict([noise_vis, X_cond_train_scaled], verbose=0)
    fake_latent = fake_latent_scaled      

    # 1. Loss & Discriminator Accuracy Curve
    plt.figure(figsize=(8,4))
    plt.plot(d_losses_hist, label='Discriminator Loss')
    plt.plot(g_losses_hist, label='Generator Loss')
    plt.title('CGAN Loss Curves')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('edinburgh_cgan_loss_curves.png')

    plt.figure(figsize=(8,4))
    plt.plot(d_accs_hist, label='Discriminator Accuracy')
    plt.title('Discriminator Accuracy over Training')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('edinburgh_cgan_discriminator_accuracy.png')

    # 2. 1D distribution compare (latent space first 3 dimensions)
    real_lat = X_lat_train_cgan_input  # Real latent noise input
    for dim in range(min(3, real_lat.shape[1])):
        rv = real_lat[:, dim]
        fv = fake_latent[:, dim]
        kde_r = gaussian_kde(rv)
        kde_f = gaussian_kde(fv)
        xs = np.linspace(min(rv.min(), fv.min()), max(rv.max(), fv.max()), 200)

        plt.figure(figsize=(6,3))
        plt.plot(xs, kde_r(xs), label='Real', linewidth=2)
        plt.plot(xs, kde_f(xs), label='Fake', linestyle='--', linewidth=2)
        plt.title(f'Latent Dimension {dim} Distribution')
        plt.xlabel(f'Latent[{dim}]')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'edinburgh_cgan_latent_dimension_{dim}_distribution.png')

    # 3. t-SNE latent space visualizing
    emb = TSNE(n_components=2, random_state=0).fit_transform(
        np.vstack([real_lat, fake_latent])
    )
    N = real_lat.shape[0]

    plt.figure(figsize=(6,6))
    plt.scatter(emb[:N,0],   emb[:N,1],   alpha=0.5, label='Real')
    plt.scatter(emb[N:,0],   emb[N:,1],   alpha=0.5, label='Fake')
    plt.legend()
    plt.title('t-SNE: Real vs. Fake Latents')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()      
    plt.savefig('edinburgh_cgan_tsne_real_vs_fake_latents.png')

    print("\n=== CGAN Evaluation Metrics ===")

    # real / fake latent vectors and condition vectors prepared
    real_latent = X_lat_test_scaled        # (M, latent_dim_cgan)
    real_cond   = X_cond_test_scaled       # (M, cond_dim)
    noise_eval  = np.random.normal(
                      0, 1, (len(real_latent), noise_dim)
                  )
    fake_latent_scaled = G_lat.predict(
                             [noise_eval, real_cond],
                             verbose=0
                         )
    # unscale
    fake_latent = scaler_lat.inverse_transform(fake_latent_scaled)
    real_lat    = scaler_lat.inverse_transform(real_latent)

    # 1) FID
    fid_val = compute_fid(real_lat, fake_latent)
    # 2) Inception Score
    is_val  = compute_inception_score(fake_latent, n_splits=10)
    # 3) FJD
    fjd_val = compute_fjd(real_lat, real_cond, fake_latent, real_cond)

    print(f"  ▶ FID: {fid_val:.4f}")
    print(f"  ▶ Inception Score: {is_val:.4f}")
    print(f"  ▶ FJD: {fjd_val:.4f}")

    wd_per_dim = [
        wasserstein_distance(real_lat[:, i], fake_latent[:, i]) 
        for i in range(real_lat.shape[1])
    ]
    wd_mean = np.mean(wd_per_dim)
    print(f"  ▶ Wasserstein Distance (mean over dims): {wd_mean:.4f}")

    wd_flatten = wasserstein_distance(real_lat.flatten(), fake_latent.flatten())
    print(f"  ▶ Wasserstein Distance (flattened): {wd_flatten:.4f}")

    real_vecs = real_lat
    fake_vecs = fake_latent

    gamma = 1.0 / real_vecs.shape[1]

    mmd_per_dim = [
        compute_mmd_rkhs(real_vecs[:, [i]], fake_vecs[:, [i]], gamma=gamma, unbiased=True)
        for i in range(real_vecs.shape[1])
    ]

    mmd_overall = compute_mmd_rkhs(real_vecs, fake_vecs, gamma=gamma, unbiased=True)

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(mmd_per_dim)), mmd_per_dim, alpha=0.7, label='Per-dimension MMD')
    plt.axhline(mmd_overall, color='red', linestyle='--',
                label=f'Overall MMD = {mmd_overall:.4f}', linewidth=2)
    plt.title('Latent Space RKHS MMD (RBF kernel, unbiased)')  
    plt.xlabel('Latent Dimension Index')  
    plt.ylabel('MMD')  
    plt.legend()  
    plt.tight_layout()  
    plt.savefig('edinburgh_cgan_rkhs_mmd_per_dimension.png')

    D = real_vecs.shape[1]

    wd_per_dim = [
        wasserstein_distance(real_vecs[:, i], fake_vecs[:, i])
        for i in range(D)
    ]
    wd_mean = np.mean(wd_per_dim)

    plt.figure(figsize=(12, 4))
    plt.bar(range(D), wd_per_dim, alpha=0.7, label='Per-dimension WD')
    plt.axhline(wd_mean, color='green', linestyle='--',
                label=f'Mean WD = {wd_mean:.4f}', linewidth=2)
    plt.title('Latent Space 1D Wasserstein Distance per Dimension')
    plt.xlabel('Latent Dimension Index')
    plt.ylabel('Wasserstein Distance')
    plt.legend()
    plt.tight_layout()
    plt.savefig('edinburgh_cgan_wasserstein_distance_per_dimension.png')
    
    for dim in range(min(3, D)):
        rv = real_vecs[:, dim]
        fv = fake_vecs[:, dim]
        kde_r = gaussian_kde(rv)
        kde_f = gaussian_kde(fv)
        xs = np.linspace(min(rv.min(), fv.min()), max(rv.max(), fv.max()), 200)
    
        plt.figure(figsize=(6, 3))
        plt.plot(xs, kde_r(xs), label='Real', linewidth=2)
        plt.plot(xs, kde_f(xs), label='Fake', linestyle='--', linewidth=2)
        plt.title(f'Latent Dimension {dim} Distribution\n1D Wasserstein = {wd_per_dim[dim]:.4f}')
        plt.xlabel(f'Latent[{dim}] Value')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'edinburgh_cgan_latent_dimension_{dim}_distribution.png')

    # --- 1. 5-Fold Cross-Validation check stability of evaluation metrics ---
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fid_scores, is_scores, fjd_scores = [], [], []
    for _, test_idx in kf.split(X_lat):
        real_lat_cv  = X_lat[test_idx]
        real_cond_cv = X_cond[test_idx]
        z_cv = np.random.normal(0,1,(len(test_idx), noise_dim))
        fake_scaled = G_lat.predict([z_cv, scaler_cond.transform(real_cond_cv)], verbose=0)
        fake_lat_cv = scaler_lat.inverse_transform(fake_scaled)
        # 지표 계산
        fid_scores.append(compute_fid(real_lat_cv, fake_lat_cv))
        is_scores.append(compute_inception_score(fake_lat_cv, n_splits=10))
        fjd_scores.append(compute_fjd(real_lat_cv, real_cond_cv, fake_lat_cv, real_cond_cv))
    print(f"  ▶ CV FID: {np.mean(fid_scores):.4f} ± {np.std(fid_scores):.4f}")
    print(f"  ▶ CV IS:  {np.mean(is_scores):.4f} ± {np.std(is_scores):.4f}")
    print(f"  ▶ CV FJD: {np.mean(fjd_scores):.4f} ± {np.std(fjd_scores):.4f}")

    # ===============================================================
    # --- 2. GAN-train & GAN-test: Generated / Real data distribution accuracy calculate ---
    # ===============================================================
    from keras.models import Sequential
    from keras.layers import Dense, Dropout

    def build_simple_classifier(input_dim):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    print("\n  ▶ Evaluating GAN-train (Train on Fake, Test on Real)...")
    X_gen = fake_latent.copy()
    # Label: Fake(0), Real(1)
    y_gen = np.zeros(len(X_gen))
    X_real = real_lat.copy()
    y_real = np.ones(len(X_real))

    X_train_gt = np.vstack([X_gen, X_real])
    y_train_gt = np.hstack([y_gen, y_real])
    p = np.random.permutation(len(X_train_gt))
    X_train_gt, y_train_gt = X_train_gt[p], y_train_gt[p]

    clf_gt = build_simple_classifier(latent_dim_cgan)
    clf_gt.fit(X_train_gt, y_train_gt, epochs=20, batch_size=64, verbose=0)
    loss_gt, acc_gt = clf_gt.evaluate(X_real, y_real, verbose=0) # 진짜를 얼마나 잘 '진짜'라고 하는가
    print(f"  ▶ GAN-train accuracy (ability to distinguish real from fake): {acc_gt:.4f}")

    # ==========================================================
    # [★★] Additional Validation (Overfitting & Stability Check)
    # ==========================================================
    print("\n=== Additional Validation (Overfitting & Stability) ===")
    
    try:
        print("\n[1] Visual Inspection with t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
        combined_latents = np.vstack([real_lat, fake_latent])
        
        tsne_results = tsne.fit_transform(combined_latents)
        
        num_real_samples = real_lat.shape[0]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(tsne_results[:num_real_samples, 0], tsne_results[:num_real_samples, 1], alpha=0.5, label='Real', s=10)
        plt.scatter(tsne_results[num_real_samples:, 0], tsne_results[num_real_samples:, 1], alpha=0.5, label='Fake', s=10)
        plt.legend()
        plt.title('t-SNE: Real vs. Fake Latents (Validation)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.tight_layout()
        plt.savefig('edinburgh_cgan_tsne_final_validation.png')
        print("  ▶ t-SNE plot saved as 'edinburgh_cgan_tsne_final_validation.png'")
        print("  ▶ Checkpoint: Real(blue) and Fake(orange) points should be well-mixed.")

        print("\n[2] Nearest Neighbor Distance Analysis (Checking for Overfitting)...")
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(real_lat)
        
        real_nn_dist, _ = nn_model.kneighbors(real_lat)        # real→real (self)
        fake_nn_dist, _ = nn_model.kneighbors(fake_latent)     # fake→real
        avg_nn_distance = np.mean(fake_nn_dist)
        print(f"  ▶ Average distance from Fake to nearest Real sample: {avg_nn_distance:.4f}")

        # 2.1 Jensen–Shannon Divergence (NN-distance distribution difference)
        from scipy.spatial.distance import jensenshannon
        from scipy.stats         import ks_2samp
        # scipy.spatial.distance.jensenshannon(signature: (p, q, base)) used
        jsd_val = jensenshannon(
            real_nn_dist.flatten(),
            fake_nn_dist.flatten(),
            base=2
        )
        print(f"  ▶ Jensen–Shannon Divergence (NN-distance): {jsd_val:.4f}")

        def rbf_kernel(X, Y, gamma):
            """RBF kernel: k(x,y)=exp(-gamma * ||x-y||^2)"""
            X_norm = np.sum(X**2, axis=1)[:, None]
            Y_norm = np.sum(Y**2, axis=1)[None, :]
            sq_dists = X_norm + Y_norm - 2 * np.dot(X, Y.T)
            return np.exp(-gamma * sq_dists)

        def compute_mmd_rkhs(X, Y, gamma=None, unbiased=True):
            """
            RKHS based MMD calculation (RBF kernel, unbiased estimation)
            - X: (n, d) actual latent vectors
            - Y: (m, d) generated latent vectors
            """
            n, m = X.shape[0], Y.shape[0]
            if gamma is None:
                gamma = 1.0 / X.shape[1]

            Kxx = rbf_kernel(X, X, gamma)
            Kyy = rbf_kernel(Y, Y, gamma)
            Kxy = rbf_kernel(X, Y, gamma)

            if unbiased:
                np.fill_diagonal(Kxx, 0)
                np.fill_diagonal(Kyy, 0)
                mmd2 = (Kxx.sum() / (n*(n-1))
                        + Kyy.sum() / (m*(m-1))
                        - 2 * Kxy.sum() / (n*m))
            else:
                mmd2 = Kxx.mean() + Kyy.mean() - 2*Kxy.mean()
            return np.sqrt(max(mmd2, 0.0))   

        def compute_mmd(x, y, gamma=None):
            if gamma is None:
                gamma = 1.0 / x.shape[1]
            def rbf(a, b):
                sq = np.sum((a[:, None, :] - b[None, :, :])**2, axis=2)
                return np.exp(-gamma * sq)
            Kxx, Kyy, Kxy = rbf(x, x), rbf(y, y), rbf(x, y)
            return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
        gamma = 1.0 / real_lat.shape[1]
        mmd_val = compute_mmd_rkhs(real_lat, fake_latent, gamma=gamma, unbiased=True)
        print(f"  ▶ Maximum Mean Discrepancy (RKHS): {mmd_val:.4f}")

        if avg_nn_distance < 0.01:
            print("  ▶ Warning: Distance is very small. The model might be overfitting (memorizing samples).")
        else:
            print("  ▶ Checkpoint: Distance is not excessively small, suggesting good generalization.")

        print("\n[3] Recalculating FID with a Larger Sample Size...")
        
        sample_size = min(5000, X_lat_train_cgan_input.shape[0])
        print(f"  ▶ Using {sample_size} samples for more stable FID calculation.")

        real_samples_large_scaled = X_lat_train_cgan_input[:sample_size]
        real_samples_large = scaler_lat.inverse_transform(real_samples_large_scaled)

        noise_large = np.random.normal(0, 1, (sample_size, noise_dim))
        cond_samples_large_scaled = X_cond_train_scaled[:sample_size]
        fake_samples_large_scaled = G_lat.predict([noise_large, cond_samples_large_scaled], verbose=0)
        fake_samples_large = scaler_lat.inverse_transform(fake_samples_large_scaled)

        fid_large_sample = compute_fid(real_samples_large, fake_samples_large)
        print(f"  ▶ FID with {sample_size} samples: {fid_large_sample:.4f}")
        print("  ▶ Checkpoint: This score provides a more reliable measure of generation quality.")

    except Exception as e:
        print(f"\n  ▶ An error occurred during additional validation: {e}")

else:
    print("CGAN training skipped or failed. No validation performed.")

fid_threshold = 0.8
if fid_val > fid_threshold:
    print(f"\n[Warning] FID is too high (<{fid_threshold}). Additional validation is performed.")

    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        tsne = TSNE(n_components=2, random_state=0)
        emb = tsne.fit_transform(np.vstack([real_lat, fake_latent]))
        plt.figure(figsize=(6,6))
        plt.scatter(emb[:len(real_lat),0], emb[:len(real_lat),1], alpha=0.5, label='Real')
        plt.scatter(emb[len(real_lat):,0], emb[len(real_lat):,1], alpha=0.5, label='Fake')
        plt.legend()
        plt.title('t-SNE: Real vs Fake Latents')
        plt.tight_layout()
        plt.savefig('edinburgh_cgan_tsne_validation.png')
        plt.close()
        print("  ▶ t-SNE visualization results saved as 'edinburgh_cgan_tsne_validation.png'.")
    except Exception as e:
        print(f"  ▶ Error during visualization validation: {e}")

    try:
        extra_n = min(5000, X_lat_train_scaled.shape[0])
        noise_extra = np.random.normal(0,1,(extra_n, noise_dim))
        cond_extra = scaler_cond.transform(X_cond[test_idx][:extra_n])
        fake_extra_scaled = G_lat.predict([noise_extra, cond_extra], verbose=0)
        fake_extra = scaler_lat.inverse_transform(fake_extra_scaled)
        real_extra = real_lat[:extra_n]
        fid_extra = compute_fid(real_extra, fake_extra)
        print(f"  ▶ Recalculated FID ({extra_n} samples): {fid_extra:.4f}")
    except Exception as e:
        print(f"  ▶ Error during FID recalculation: {e}")

    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=1).fit(real_lat)
        dists, _ = nn.kneighbors(fake_latent)
        avg_nn = np.mean(dists)
        print(f"  ▶ Fake→Real Average NN Distance: {avg_nn:.4f}")
        if avg_nn < 0.01:
            print("    -> Warning: Generated samples are very similar to real samples. Overfitting possible.")
    except Exception as e:
        print(f"  ▶ Error during overfitting check: {e}")
  

# -----------------------------------------------
# 17. Similar user and preference prediction functions
# -----------------------------------------------

def find_similar_users_by_pref(target_user_id, sim_model, user_ids_knn_list, top_k=5):
    if target_user_id not in user_ids_knn_list or sim_model is None:
        return []
    
    target_user_idx = user_ids_knn_list.index(target_user_id)
    target_vector = user_pref_df_cleaned.iloc[target_user_idx].values.reshape(1, -1)
    
    distances, indices = sim_model.kneighbors(target_vector, n_neighbors=top_k + 1)
    
    similar_users = []
    for i in range(1, len(indices[0])): 
        sim_user_idx = indices[0][i]
        sim_user_id = user_ids_knn_list[sim_user_idx]
        similarity = 1 - distances[0][i]
        similar_users.append((sim_user_id, similarity))
        
    return similar_users    

def evaluate_precision_ndcg_academic(user_id, recommended_routes, ground_truth_pois, K_values=[5, 10]):
    actual_pois = ground_truth_pois

    if not actual_pois:
        return {f'Precision@{k}': 0.0 for k in K_values} | {f'NDCG@{k}': 0.0 for k in K_values}

    recommended_pois = []
    for route in recommended_routes:
        recommended_pois.extend([normalize_poi_name(poi) for poi in route['sequence']])

    results = {}

    for k in K_values:
        top_k_pois = recommended_pois[:k]
        relevant_count = sum(1 for poi in top_k_pois if poi in actual_pois)
        precision_k = relevant_count / k if k > 0 else 0.0
        results[f'Precision@{k}'] = precision_k

    for k in K_values:
        dcg = 0.0
        for i, poi in enumerate(recommended_pois[:k]):
            if poi in actual_pois:
                dcg += 1.0 / np.log2(i + 2)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual_pois), k)))
        ndcg_k = dcg / idcg if idcg > 0 else 0.0
        results[f'NDCG@{k}'] = ndcg_k

    return results        

def get_user_ground_truth_pois(user_id, df, moving_cols):
    user_rows = df[df['userID'] == user_id]
    ground_truth_pois = set()
    for _, row in user_rows.iterrows():
        for col in moving_cols:
            poi = row[col]
            if pd.notnull(poi):
                ground_truth_pois.add(normalize_poi_name(poi))
    return ground_truth_pois

def calculate_standard_precision_recall_f1(recommended_routes, ground_truth_pois):
    recommended_pois = {normalize_poi_name(poi) for route in recommended_routes for poi in route.get('sequence', [])}
    if not recommended_pois or not ground_truth_pois:
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    true_positives = len(recommended_pois.intersection(ground_truth_pois))
    precision = true_positives / len(recommended_pois)
    recall = true_positives / len(ground_truth_pois)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}

# -----------------------------------------------
# 18. Recommended function: “CGAN Latent → POI sequence”
# -----------------------------------------------
def recommend_routes_enhanced(target_user_id, pois_to_exclude=None, top_k_pois_per_route=3, num_gen_routes=3, lambda_mmr=0.7):
    if 'G_lat' not in globals() or not G_lat: return {}
    if target_user_id not in combined_user_emb: return {}

    enhanced_cond_vec = create_collaborative_condition_vector(target_user_id, sim_user_model, user_ids_pref_knn)
    cond_vec_scaled = scaler_cond.transform(enhanced_cond_vec.reshape(1, -1))
    noise_for_gen = np.random.normal(0, 1, (num_gen_routes, noise_dim))
    generated_latents_scaled = G_lat.predict([noise_for_gen, cond_vec_scaled.repeat(num_gen_routes, axis=0)], verbose=0)
    generated_latents = scaler_lat.inverse_transform(generated_latents_scaled)
    
    # --- Step 1: Anchor POI Selection (Cold-Start Handling) ---
    visited = user_visited_places.get(target_user_id, set())
    if not visited:
        anchor_pois = random.sample(list(valid_places), num_gen_routes)
    else:
        anchor_candidates = visited
        if pois_to_exclude:
            anchor_candidates -= pois_to_exclude

        if not anchor_candidates:
            anchor_candidates = {p for p in valid_places if p in place_emb}
            if pois_to_exclude:
                anchor_candidates -= pois_to_exclude

        anchor_pois = []
        for latent_vec in generated_latents:
            best_anchor, max_relevance = None, -1.0
            for poi in anchor_candidates:
                rel = 1 - cosine(place_emb[poi], latent_vec)
                if rel > max_relevance:
                    best_anchor, max_relevance = poi, rel
            if best_anchor:
                anchor_pois.append(best_anchor)

    # --- Step 2: Route Expansion ---
    
    final_routes = []
    used_pois_across_all_routes = set(anchor_pois)

    for i, anchor_poi in enumerate(anchor_pois):
        gen_latent = generated_latents[i]
        route_sequence = [anchor_poi]
        
        expansion_candidates = {p for p in valid_places if p in place_emb} - used_pois_across_all_routes
        
        if anchor_poi in poi_list_for_knn and poi_sim_model:
            num_neighbors_to_request = min(50, poi_sim_model.n_samples_fit_)
            
            p_idx = poi_list_for_knn.index(anchor_poi)
            if num_neighbors_to_request > 0:
                _, indices = poi_sim_model.kneighbors([poi_matrix_for_knn[p_idx]], n_neighbors=num_neighbors_to_request)
                similar_pois = {poi_list_for_knn[idx] for idx in indices.flatten()}
                expansion_candidates.update(similar_pois - used_pois_across_all_routes)

        for _ in range(top_k_pois_per_route - 1):
            if not expansion_candidates: break
            
            next_poi = None
            max_score = -1
            
            last_poi_in_route_emb = place_emb[route_sequence[-1]]
            
            for candidate_poi in expansion_candidates:
                candidate_emb = place_emb[candidate_poi]
                score = (1 - cosine(candidate_emb, gen_latent)) * 0.6 + \
                        (1 - cosine(candidate_emb, last_poi_in_route_emb)) * 0.4

                if score > max_score:
                    max_score = score
                    next_poi = candidate_poi
            
            if next_poi:
                route_sequence.append(next_poi)
                used_pois_across_all_routes.add(next_poi)
                expansion_candidates.remove(next_poi)

        if route_sequence:
            optimized_sequence = optimize_route_tsp(route_sequence, place_to_coord)
            final_routes.append({
                'routeID': f"Anchor-Expand_Route_{target_user_id}_{i+1}",
                'sequence': optimized_sequence,
                'themes': list(set(th for p in optimized_sequence if p in place_themes for th in place_themes[p])),
                'total_distance': compute_route_distance(optimized_sequence, place_to_coord, target_user_id, user_visited_places, df, moving_cols)
            })

    user_rows = df[df['userID'] == target_user_id]
    last_poi_for_context = None
    if not user_rows.empty:
        for _, row in user_rows.iloc[::-1].iterrows():
            for col in reversed(moving_cols):
                val = row[col]
                if pd.notnull(val) and val in place_emb:
                    last_poi_for_context = val
                    break
            if last_poi_for_context: break

    return {'user_id': target_user_id, 'recommended_routes': final_routes, 'last_known_poi_for_context': last_poi_for_context}      

def optimize_route_tsp(poi_names, place_to_coord):
    coords = [place_to_coord[p] for p in poi_names if p in place_to_coord]
    n = len(coords)
    if n > 6:
        return poi_names
    min_dist = float('inf')
    best_seq = poi_names
    for perm in permutations(range(n)):
        route = [coords[i] for i in perm]
        dist = sum(
            haversine(route[i][0], route[i][1], route[i+1][0], route[i+1][1])
            for i in range(n-1)
        )
        if dist < min_dist:
            min_dist = dist
            best_seq = [poi_names[i] for i in perm]
    return best_seq

def reorder_route_start_nearest_last_poi(pois, place_to_coord, last_poi):
    if not pois or not last_poi or last_poi not in place_to_coord:
        return pois

    last_poi_coord = place_to_coord[last_poi]

    def get_dist_from_last(poi):
        if poi in place_to_coord:
            return haversine(last_poi_coord[0], last_poi_coord[1],
                             place_to_coord[poi][0], place_to_coord[poi][1])
        return float('inf')

    pois.sort(key=get_dist_from_last)
    return pois


def plot_cgan_recommended_routes_on_folium(recs, place_to_coord, last_poi, filepath='cgan_recommended_route_map.html'):

    all_coords, all_place_names, route_pois = [], [], []
    optimized_route_pois = []

    for entry in recs.get('recommended_routes', []):
        pois = [p for p in entry['sequence'] if p in place_to_coord]
        if pois:
            pois = reorder_route_start_nearest_last_poi(pois, place_to_coord, last_poi)
        if pois:
            first = pois[0]
            rest = pois[1:]
            rest_opt = optimize_route_tsp(rest, place_to_coord) if rest else []
            pois = [first] + rest_opt
        optimized_route_pois.append(pois)
        for place in pois:
            if place not in all_place_names:
                all_coords.append(place_to_coord[place])
                all_place_names.append(place)

    if last_poi and last_poi in place_to_coord:
        if last_poi not in all_place_names:
            all_coords.append(place_to_coord[last_poi])
            all_place_names.append(last_poi)

    if not all_coords:
        print("No recommended routes found.")
        return

    center = [np.mean([c[0] for c in all_coords]), np.mean([c[1] for c in all_coords])]
    m = folium.Map(location=center, zoom_start=13)

    colors = ['blue', 'red', 'green', 'purple', 'orange', 'gray', 'pink', 'cyan', 'yellow', 'brown']
    for route_idx, pois in enumerate(optimized_route_pois, 1):
        if not pois:
            continue
        color = colors[route_idx % len(colors)]
        group = folium.FeatureGroup(name=f"Route {route_idx} ({color})")
        for i in range(len(pois) - 1):
            p1, p2 = pois[i], pois[i + 1]
            path_coords = [place_to_coord[p1], place_to_coord[p2]]
            folium.PolyLine(
                locations=path_coords,
                color=color,
                weight=5,
                opacity=0.8,
                popup=f"Route {route_idx}: {p1} → {p2}"
            ).add_to(group)

        if last_poi and last_poi in place_to_coord and pois and pois[0] in place_to_coord:
            path_coords = [place_to_coord[last_poi], place_to_coord[pois[0]]]
            folium.PolyLine(
                locations=path_coords,
                color='black',
                weight=4,
                opacity=0.7,
                dash_array='5, 5',
                popup=f"From Last Visited ({last_poi}) to Route {route_idx} Start ({pois[0]})"
            ).add_to(group)

        group.add_to(m)

    for idx, (lat, lon) in enumerate(all_coords, 1):
        place_name = all_place_names[idx - 1]
        if place_name == last_poi:
            folium.Marker(
                [lat, lon],
                popup=f"Last Visited: {place_name}",
                tooltip="Last Visited",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)
        else:
            folium.Marker(
                [lat, lon],
                popup=f"{idx}. {place_name}",
                tooltip=str(idx),
                icon=folium.Icon(color='red', icon='star')
            ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(filepath)
    print(f"Recommended routes map saved to '{filepath}'.")

def evaluate_theme_preference_strict(user_id, recommended_routes, user_preferences, all_themes_list):
    if user_id not in user_preferences:
        return 0.0

    user_pref_vector = np.array([
        user_preferences[user_id].get(theme, 0.0) for theme in all_themes_list
    ])

    rec_theme_counts = Counter()
    total_pois = 0
    for route in recommended_routes:
        for poi in route.get('sequence', []):
            if poi in place_themes:
                total_pois += 1
                for theme in place_themes[poi]:
                    rec_theme_counts[theme] += 1
    
    if total_pois == 0:
        return 0.0

    rec_theme_vector = np.array([
        rec_theme_counts.get(theme, 0) for theme in all_themes_list
    ], dtype=float)

    user_norm = np.linalg.norm(user_pref_vector)
    rec_norm = np.linalg.norm(rec_theme_vector)

    if user_norm == 0 or rec_norm == 0:
        return 0.0
    
    cosine_similarity = np.dot(user_pref_vector, rec_theme_vector) / (user_norm * rec_norm)
    return max(0, cosine_similarity)

# def debug_route_data_structure(df, user_id):
#     print(f"\n=== Data Structure Debug for User {user_id} ===")
    
#     moving_cols = [col for col in df.columns if col.startswith('movingpath')]
#     print(f"Found {len(moving_cols)} movingpath columns: {moving_cols[:10]}...")
    
#     user_rows = df[df['userID'] == user_id]
#     print(f"User has {len(user_rows)} rows in dataset")
    
#     if not user_rows.empty:
#         first_row = user_rows.iloc[0]
#         print("Sample movingpath data:")
#         for col in moving_cols[:10]:
#             value = first_row[col] if pd.notnull(first_row[col]) else "NaN"
#             print(f"  {col}: {value}")
    
#     all_pois = set()
#     for col in moving_cols:
#         unique_vals = df[col].dropna().unique()
#         all_pois.update(unique_vals)
#     print(f"Total unique POIs in dataset: {len(all_pois)}")

# def debug_evaluation_detailed(user_id, recommended_routes, df):
#     """Detailed debugging information output"""
#     print(f"=== Detailed Debugging for User {user_id} ===")
    
#     user_rows = df[df['userID'] == user_id]
#     print(f"User has {len(user_rows)} historical records")
    
#     # User's actual visited POIs
#     user_pois = set()
#     for _, row in user_rows.iterrows():
#         for col in moving_cols:
#             if pd.notnull(row[col]):
#                 user_pois.add(row[col])
    
#     print(f"User visited {len(user_pois)} unique POIs: {list(user_pois)[:10]}...")
    
#     # Recommended POIs
#     rec_pois = set()
#     for route in recommended_routes:
#         rec_pois.update(route['sequence'])
    
#     print(f"Recommended {len(rec_pois)} unique POIs: {list(rec_pois)[:10]}...")
    
#     # Intersection
#     intersection = user_pois & rec_pois
#     print(f"Intersection: {len(intersection)} POIs: {list(intersection)}")
    
#     # Theme analysis
#     user_themes = set()
#     for poi in user_pois:
#         if poi in place_themes:
#             user_themes.update(place_themes[poi])
    
#     rec_themes = set()
#     for poi in rec_pois:
#         if poi in place_themes:
#             rec_themes.update(place_themes[poi])
    
#     print(f"User themes: {user_themes}")
#     print(f"Recommended themes: {rec_themes}")
#     print(f"Theme intersection: {user_themes & rec_themes}")

def evaluate_preference_rmse_mae(user_id, recommended_routes, user_preferences, place_themes):
    """
    Calculate the RMSE and MAE between the theme distribution of recommended routes and the actual user preferences.
    """
    if user_id not in user_preferences:
        return {'RMSE': np.nan, 'MAE': np.nan}

    actual_pref_themes = sorted(user_preferences[user_id].keys())
    actual_pref_vector = np.array([user_preferences[user_id][theme] for theme in actual_pref_themes])

    recommended_theme_counts = Counter()
    total_pois = 0
    for route in recommended_routes:
        for poi in route.get('sequence', []):
            if poi in place_themes:
                total_pois += 1
                for theme in place_themes[poi]:
                    clean_theme = theme.replace('pref_', '').replace('_', ' ')
                    recommended_theme_counts[clean_theme] += 1
    
    if total_pois == 0:
        return {'RMSE': np.nan, 'MAE': np.nan}

    predicted_pref_vector = np.zeros_like(actual_pref_vector, dtype=float)
    for i, theme in enumerate(actual_pref_themes):

        predicted_pref_vector[i] = recommended_theme_counts.get(theme, 0) / total_pois

    rmse = np.sqrt(mean_squared_error(actual_pref_vector, predicted_pref_vector))
    mae = mean_absolute_error(actual_pref_vector, predicted_pref_vector)

    return {'RMSE': rmse, 'MAE': mae}

def evaluate_pairs_f1_score(user_id: str, recommended_routes: list[dict], df: pd.DataFrame, moving_cols: list[str]) -> dict[str, float]:
    """
    Calculates the pairs-F1 score by comparing ordered POI pairs from recommended and actual routes.

    This function evaluates the sequential accuracy of trajectory recommendations based on the formula:
    pairs-F1 = (2 * P_pair * R_pair) / (P_pair + R_pair)

    Args:
        user_id: The ID of the user to evaluate.
        recommended_routes: A list of recommended routes, where each route is a dictionary
                            containing a 'sequence' of POI names.
        df: The DataFrame containing the ground truth user trajectory data.
        moving_cols: The list of column names in the DataFrame that represent the user's path.

    Returns:
        A dictionary containing the precision (P_pair), recall (R_pair),
        and F1-score (pairs_f1) for the ordered POI pairs.
    """
    # 1. Extract ordered POI pairs from the user's ground truth trajectories.
    actual_pairs = set()
    user_rows = df[df['userID'] == user_id]
    for _, row in user_rows.iterrows():
        # Create a sequence of POIs for the current trajectory
        route_sequence = [normalize_poi_name(row[col]) for col in moving_cols if pd.notnull(row[col])]
        # Generate ordered pairs (A, B) from the sequence
        if len(route_sequence) > 1:
            for i in range(len(route_sequence) - 1):
                actual_pairs.add((route_sequence[i], route_sequence[i+1]))

    # 2. Extract ordered POI pairs from the recommended routes.
    recommended_pairs = set()
    for route in recommended_routes:
        # Ensure the sequence is iterable before processing
        sequence = route.get('sequence', [])
        if isinstance(sequence, Iterable):
            rec_sequence = [normalize_poi_name(poi) for poi in sequence]
            # Generate ordered pairs from the recommended sequence
            if len(rec_sequence) > 1:
                for i in range(len(rec_sequence) - 1):
                    recommended_pairs.add((rec_sequence[i], rec_sequence[i+1]))

    # If there are no pairs in either set, the scores are 0.
    if not recommended_pairs or not actual_pairs:
        return {'P_pair': 0.0, 'R_pair': 0.0, 'pairs_f1': 0.0}

    # 3. Calculate precision and recall for the ordered pairs.
    true_positives = len(recommended_pairs.intersection(actual_pairs))

    p_pair = true_positives / len(recommended_pairs) if recommended_pairs else 0.0
    r_pair = true_positives / len(actual_pairs) if actual_pairs else 0.0

    # 4. Calculate the pairs-F1 score.
    pairs_f1 = (2 * p_pair * r_pair) / (p_pair + r_pair) if (p_pair + r_pair) > 0 else 0.0

    return {'P_pair': p_pair, 'R_pair': r_pair, 'pairs_f1': pairs_f1}

def create_user_preferences(df):
    """Create user preference dictionary from CSV data"""
    user_preferences = {}
    
    # Preference column names
    pref_columns = [col for col in df.columns if col.startswith('pref_')]
    
    for user_id in df['userID'].unique():
        user_rows = df[df['userID'] == user_id]
        
        if not user_rows.empty:
            first_row = user_rows.iloc[0]
            
            # User preference dictionary generation
            user_prefs = {}
            for col in pref_columns:
                theme_name = col.replace('pref_', '')
                preference_value = first_row[col]
                
                if pd.isna(preference_value):
                    preference_value = 0.0
                
                user_prefs[theme_name] = float(preference_value)
            
            user_preferences[user_id] = user_prefs
    
    return user_preferences

# Generate User Preference Dictionary
print("Creating user preferences dictionary...")
user_preferences = create_user_preferences(df)

print(f"Created preferences for {len(user_preferences)} users")

# 1) ─ Average Jensen-Shannon Divergence
def avg_jsd(real_vecs: np.ndarray, fake_vecs: np.ndarray, bins: int = 30) -> float:
    """real_vecs/fake_vecs  =  (N, D)  latent matrix"""
    from scipy.spatial.distance import jensenshannon
    js_vals = []
    for d in range(real_vecs.shape[1]):
        r, f = real_vecs[:, d], fake_vecs[:, d]
        all_v = np.concatenate([r, f])
        hist_range = (all_v.min(), all_v.max())
        r_hist, _ = np.histogram(r, bins=bins, range=hist_range, density=True)
        f_hist, _ = np.histogram(f, bins=bins, range=hist_range, density=True)
        r_hist += 1e-12
        f_hist += 1e-12
        js_vals.append(jensenshannon(r_hist, f_hist, base=2))
    return float(np.mean(js_vals))

# 2) ─ Average 1-D Wasserstein Distance (이미 정의된 함수 활용)
def avg_wasserstein(real_vecs: np.ndarray, fake_vecs: np.ndarray) -> float:
    return compute_wasserstein_distance(real_vecs, fake_vecs)   # 앞서 정의됨

# 3) ─ Difference in pair-wise correlation
def diff_pairwise_corr(real_vecs: np.ndarray, fake_vecs: np.ndarray) -> float:
    real_corr = np.corrcoef(real_vecs, rowvar=False)
    fake_corr = np.corrcoef(fake_vecs, rowvar=False)
    iu = np.triu_indices_from(real_corr, k=1)
    diff = np.abs(real_corr[iu] - fake_corr[iu])
    return float(np.mean(diff))

# 4) ─ AUC 
def compute_auc(real_lat: np.ndarray,
                fake_lat: np.ndarray,
                disc_model: Model,
                cond_vecs: np.ndarray) -> float:
    y_true = np.concatenate([np.ones(len(real_lat)),
                             np.zeros(len(fake_lat))])
    real_cond = np.repeat(cond_vecs[:1], len(real_lat), axis=0)
    fake_cond = np.repeat(cond_vecs[:1], len(fake_lat), axis=0)
    y_score_real = disc_model.predict([real_lat, real_cond], verbose=0).ravel()
    y_score_fake = disc_model.predict([fake_lat, fake_cond], verbose=0).ravel()
    y_score = np.concatenate([y_score_real, y_score_fake])
    return float(roc_auc_score(y_true, y_score))

# ──────────────────────────────────────────────
# 5)  Calculation (real_lat, fake_latent, real_cond already obtained)
# ──────────────────────────────────────────────
avg_jsd_val  = avg_jsd(real_lat,     fake_latent)
avg_wd_val   = avg_wasserstein(real_lat, fake_latent)
diff_corr_val= diff_pairwise_corr(real_lat, fake_latent)
auc_val      = compute_auc(real_lat, fake_latent, D_lat, real_cond)

# ──────────────────────────────────────────────
# 6)  Results
# ──────────────────────────────────────────────
metrics_tbl = pd.DataFrame(
    {
        "Metric" : ["Average JSD",
                    "Average Wasserstein Distance",
                    "Diff. pair-wise Correlation",
                    "AUC"],
        "Value"  : [avg_jsd_val,
                    avg_wd_val,
                    diff_corr_val,
                    auc_val]
    }
)

# Table for evaluation results
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(metrics_tbl.to_string(index=False))
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")


print("--- Pre-filtering users for evaluation ---")
MIN_TRIPS_FOR_EVAL = 3
evaluatable_users = [uid for uid in df['userID'].unique()
                     if len(df[df['userID'] == uid]) >= MIN_TRIPS_FOR_EVAL]
print(f"Found {len(evaluatable_users)} users suitable for evaluation.")

sample_user_ids = evaluatable_users
print(f"\n--- Starting evaluation for all {len(sample_user_ids)} users: {sample_user_ids} ---")

if __name__ == '__main__':
    print(f"\n--- Script Setup Complete. Time: {time.time() - script_start:.2f} seconds ---")

    NUM_EVAL_USERS = 5 # Number of users to evaluate
    # -------------------

    # Filter users with suitable number of trips (Minimum 3 POIs)
    MIN_TRIPS_FOR_EVAL = 3
    evaluatable_users = [uid for uid in df['userID'].unique() if len(df[df['userID'] == uid]) >= MIN_TRIPS_FOR_EVAL]
    
    if not evaluatable_users:
        print("No users suitable for evaluation. (Minimum 3 POIs required)")
    else:
        sample_user_ids = evaluatable_users
        print(f"\n--- Starting evaluation for {len(sample_user_ids)} users ---")

        all_metrics = defaultdict(list)
    
        for user_id in sample_user_ids:
            print("\n" + "="*80)
            print(f"--- Processing user: {user_id} ---")

            ground_truth_pois = get_user_ground_truth_pois(user_id, df, moving_cols)

            if not ground_truth_pois:
                print("    ▶ Valid ground truth could not be constructed for this user. Skipping.")
                continue

            recommendations = recommend_routes_enhanced(
                user_id,
                pois_to_exclude=None
            )

            if not recommendations or not recommendations.get('recommended_routes'):
                print("    ▶ No recommendations could be generated for this user.")
                continue

            print("\n  [Generated Routes]")
            for i, route in enumerate(recommendations['recommended_routes']):
                seq_str = " → ".join(map(str, route['sequence']))
                print(f"    Route {i+1}: {seq_str}")

            map_filepath = f"cgan_recommended_route_map_{user_id}.html"
            plot_cgan_recommended_routes_on_folium(recommendations, place_to_coord, recommendations.get('last_known_poi_for_context'), filepath=map_filepath)
            print(f"\n  ▶ Route map saved to '{map_filepath}'")

            # --- Evaluation Metrics ---
            print("\n  [Evaluation Metrics]")
            
            # POI-based standard Precision, Recall, F1-Score calculation
            standard_metrics = calculate_standard_precision_recall_f1(
                recommendations['recommended_routes'],
                ground_truth_pois
            )
            all_metrics['precision'].append(standard_metrics['precision'])
            all_metrics['recall'].append(standard_metrics['recall'])
            all_metrics['f1_score'].append(standard_metrics['f1_score'])

            print(f"    ▶ POI Precision: {standard_metrics['precision']:.4f}")
            print(f"    ▶ POI Recall:    {standard_metrics['recall']:.4f}")
            print(f"    ▶ POI F1-Score:  {standard_metrics['f1_score']:.4f}")
            
            precision_ndcg_results = evaluate_precision_ndcg_academic(
                user_id,
                recommendations['recommended_routes'],
                ground_truth_pois,
                K_values=[5, 10]
            )
            for k_val in [5, 10]:
                all_metrics[f'precision@{k_val}'].append(precision_ndcg_results[f'Precision@{k_val}'])
                all_metrics[f'ndcg@{k_val}'].append(precision_ndcg_results[f'NDCG@{k_val}'])

            # print(f"    ▶ Precision@5:   {precision_ndcg_results['Precision@5']:.4f}, NDCG@5:   {precision_ndcg_results['NDCG@5']:.4f}")
            # print(f"    ▶ Precision@10:  {precision_ndcg_results['Precision@10']:.4f}, NDCG@10:  {precision_ndcg_results['NDCG@10']:.4f}")

            theme_accuracy = evaluate_theme_preference_strict(
                user_id,
                recommendations['recommended_routes'],
                user_preferences,
                all_themes
            )
            all_metrics['theme_profile_accuracy'].append(theme_accuracy)
            print(f"    ▶ Theme Profile Match (Cosine): {theme_accuracy:.4f}")

            rmse_mae_results = evaluate_preference_rmse_mae(
                user_id,
                recommendations['recommended_routes'],
                user_preferences,
                place_themes
            )
            if not np.isnan(rmse_mae_results['RMSE']):
                all_metrics['rmse'].append(rmse_mae_results['RMSE'])
            if not np.isnan(rmse_mae_results['MAE']):
                all_metrics['mae'].append(rmse_mae_results['MAE'])

            print(f"    ▶ Theme Profile RMSE: {rmse_mae_results['RMSE']:.4f}, MAE: {rmse_mae_results['MAE']:.4f}")

            # =========================================================================
            # Pairs-F1 score calculation and output
            # =========================================================================
            pairs_f1_results = evaluate_pairs_f1_score(
                user_id,
                recommendations['recommended_routes'],
                df,
                moving_cols
            )
            all_metrics['P_pair'].append(pairs_f1_results['P_pair'])
            all_metrics['R_pair'].append(pairs_f1_results['R_pair'])
            all_metrics['pairs_f1'].append(pairs_f1_results['pairs_f1'])
            
            print(f"    ▶ Pairs Precision (P_pair): {pairs_f1_results['P_pair']:.4f}")
            print(f"    ▶ Pairs Recall (R_pair):    {pairs_f1_results['R_pair']:.4f}")
            print(f"    ▶ Pairs F1-Score:           {pairs_f1_results['pairs_f1']:.4f}")
            # =========================================================================

        # --- Final summary output ---
        print("\n" + "="*80)
        print("--- Overall Evaluation Summary ---")
        print(f"Total users processed: {len(sample_user_ids)}")

        summary_metrics = {
            "Precision": "precision",
            "Recall": "recall",
            "F1-Score": "f1_score",
            "Theme Profile Accuracy (Cosine)": "theme_profile_accuracy",
            "Theme Profile RMSE": "rmse",
            "Theme Profile MAE": "mae",
            "Pairs F1-Score": "pairs_f1"
        }

        for display_name, metric_key in summary_metrics.items():
            values = all_metrics.get(metric_key, [])
            if values:
                avg_value = np.mean(values)
                std_dev = np.std(values)
                print(f"  ▶ Avg {display_name}: {avg_value:.4f} ± {std_dev:.4f}")
            else:
                print(f"  ▶ Avg {display_name}: N/A")

        print("\n\n" + "─"*20 + " POI count (k) sensitivity analysis " + "─"*20)
        
        k_values_to_test = [3, 4, 5, 6]
        
        ks, fid_scores = evaluate_performance_by_k_with_fid(
            k_range=k_values_to_test,
            num_gen_routes_per_k=100
        )
        
        valid_indices = ~np.isnan(fid_scores)
        ks_valid = np.array(ks)[valid_indices]
        fids_valid = np.array(fid_scores)[valid_indices]
        
        if len(ks_valid) > 0:
            plt.figure(figsize=(8, 6))
            plt.plot(ks_valid, fids_valid, marker='o', linestyle='-', markersize=8, linewidth=2, color='purple')
            
            plt.xlabel('Number of POIs in recommended route (k)')
            plt.ylabel('Unified performance score (FID, lower is better)')
            plt.title('POI count (k) sensitivity analysis')
            plt.xticks(k_values_to_test)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend()
            plt.tight_layout()
            
            sensitivity_plot_path = 'edinburgh_poi_performance_analysis_unified.png'
            plt.savefig(sensitivity_plot_path)
            print(f"\n▶ Unified performance analysis graph saved to '{sensitivity_plot_path}'")
            plt.show()
        else:
            print("▶ Not enough valid data to generate graph.")
                
        print("="*80)

# ─────────────────────────────────────────────────────────────────────
# ▶ Cold-start User Test: userID=111111
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_user = '111111'
    print(f"\n=== Cold-start Test for User {test_user} ===")
    recs = recommend_routes_enhanced(
        test_user,
        pois_to_exclude=None,
        top_k_pois_per_route=3,
        num_gen_routes=3
    )
    routes = recs.get('recommended_routes', [])
    if not routes:
        print("▶ Cannot generate routes.")
    else:
        # --- Point all routes in a single map ---
        first_pts = [
            place_to_coord[r['sequence'][0]]
            for r in routes
            if r['sequence'] and r['sequence'][0] in place_to_coord
        ]
        if first_pts:
            center_lat = sum(lat for lat, _ in first_pts) / len(first_pts)
            center_lng = sum(lng for _, lng in first_pts) / len(first_pts)
        else:
            # 마지막 방문 POI의 중심 혹은 기본 (0,0) fallback
            center = get_user_geo_center(test_user) or (0, 0)
            center_lat, center_lng = center

        m = folium.Map(location=[center_lat, center_lng], zoom_start=13)
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for idx, route in enumerate(routes, 1):
            seq = route['sequence']
            seq_str = " → ".join(seq)
            print(f"▶ Route {idx}: {seq_str}")
            coords = [
                place_to_coord[p]
                for p in seq
                if p in place_to_coord
            ]
            if not coords:
                print(f"   ▶ Route {idx}: Insufficient coordinates to display on map")
                continue
            color = colors[(idx-1) % len(colors)]
            folium.PolyLine(coords,
                            color=color,
                            weight=4,
                            opacity=0.8,
                            tooltip=f"Route {idx}"
            ).add_to(m)
            for poi, (lat, lng) in zip(seq, coords):
                folium.Marker(
                    location=(lat, lng),
                    popup=f"Route {idx}: {poi}",
                    icon=folium.Icon(color=color)
                ).add_to(m)
        map_filename = f"edinburgh_routes_{test_user}.html"
        m.save(map_filename)
        print(f"   ▶ All routes saved to '{map_filename}'")
