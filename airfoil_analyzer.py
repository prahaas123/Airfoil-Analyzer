import pandas as pd
import numpy as np
import os
import requests
import re
import json
import streamlit as st
import neuralfoil as nf
import aerosandbox as asb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def fetch_af_polar(af_name, re_num):
    print(f"  -> COMPUTING: {af_name} at Re={re_num} using NeuralFoil...")
    alphas = np.linspace(-5, 15, 101)
    try:
        aero = nf.get_aero_from_airfoil(
            airfoil=asb.Airfoil(name=af_name),
            alpha=alphas,
            Re=re_num
        )
        
        cl = aero['CL']
        cd = aero['CD']
        cm = aero['CM']
        
    except Exception as e:
        print(f"  -> ERROR: Failed to compute {af_name} with NeuralFoil: {e}")
        return (None, None, None, None), (None, None)

    ld = cl / cd
    cl_max = np.max(cl)
    ld_max = np.max(ld)
    
    alpha_stall = alphas[np.argmax(cl)]
    alpha_ld_max = alphas[np.argmax(ld)]
    
    return (alphas, cl, cd, cm), [cl_max, ld_max]

def search_airfoils_by_geometry(min_thick=0.0, max_thick=66.4, min_camber=0.0, max_camber=16.4, json_filename="airfoils.json"):
    print(f"Searching '{json_filename}' for Thickness: {min_thick}-{max_thick}% | Camber: {min_camber}-{max_camber}%...")
    airfoil_names = []
    
    try:
        with open(json_filename, 'r', encoding='utf-8') as f:
            airfoil_data = json.load(f)
            
        for name, specs in airfoil_data.items():                
            thickness = specs.get("max_thickness_percent")
            camber = specs.get("max_camber_percent")
            if thickness is not None and camber is not None:
                if (min_thick <= thickness <= max_thick) and (min_camber <= camber <= max_camber):
                    airfoil_names.append(name)
                    
        print(f"  -> Found {len(airfoil_names)} airfoils matching the criteria.")
        return airfoil_names
        
    except FileNotFoundError:
        print(f"[ERROR] Could not find '{json_filename}'. Please ensure the file exists in the current directory.")
        return []
    except json.JSONDecodeError:
        print(f"[ERROR] '{json_filename}' is corrupted or not a valid JSON file.")
        return []
    
def search_airfoils_by_name(*search_terms, json_filename="airfoils.json"):
    if not search_terms:
        print("  -> No search terms provided.")
        return []

    print(f"Searching '{json_filename}' for terms: {', '.join(search_terms)}...")
    matched_airfoils = []
    terms_lower = [term.lower() for term in search_terms]
    
    try:
        with open(json_filename, 'r', encoding='utf-8') as f:
            airfoil_data = json.load(f)
            
        for name in airfoil_data.keys():
            name_lower = name.lower()
            if any(term in name_lower for term in terms_lower):
                matched_airfoils.append(name)
                
        print(f"  -> Found {len(matched_airfoils)} airfoils matching the search terms.")
        return matched_airfoils
        
    except FileNotFoundError:
        print(f"[ERROR] Could not find '{json_filename}'. Please ensure the file exists in the current directory.")
        return []
    except json.JSONDecodeError:
        print(f"[ERROR] '{json_filename}' is corrupted or not a valid JSON file.")
        return []

def filter_cm_at_alpha(airfoils, properties, target_alpha, min_cm, max_cm):
    passing_names = set()
    for name, (alphas, cl, cd, cm) in airfoils.items():
        cm_at_target = np.interp(target_alpha, alphas, cm)
        if min_cm <= cm_at_target <= max_cm:
            passing_names.add(name)

    filtered_airfoils = {n: v for n, v in airfoils.items() if n in passing_names}
    filtered_properties = [row for row in properties if row[0] in passing_names]

    print(f"\n  -> {len(filtered_airfoils)} / {len(airfoils)} airfoils passed the Cm filter "
          f"({min_cm} <= Cm <= {max_cm} at alpha={target_alpha}°).")

    return filtered_airfoils, filtered_properties

def filter_top_clmax(airfoils, properties, top_n= 100):
    sorted_properties = sorted(properties, key=lambda x: x[1], reverse=True)
    top_properties = sorted_properties[:top_n]
    top_names = {prop[0] for prop in top_properties}
    filtered_airfoils = {name: data for name, data in airfoils.items() if name in top_names}
    print(f"\n  -> Filtered down to the top {len(filtered_airfoils)} airfoils by Cl_max.")
    return filtered_airfoils, top_properties

st.set_page_config(page_title="Airfoil Analyzer", layout="wide")
st.title("Airfoil Analyzer")

# Data Fetching Functions (Cached)
@st.cache_data
def load_and_compute_data(reynolds_number=100000):
    airfoils = {}
    properties = []
    reynolds_number = 100000
    
    for airfoil in search_airfoils_by_name("hq", "goe", "sd", "sg", "n-11", "clark", "ham"):
        (alphas, cl, cd, cm), (clmax, ldmax) = fetch_af_polar(airfoil, reynolds_number)
        if alphas is not None:
            airfoils[airfoil] = alphas, cl , cd, cm
            properties.append([airfoil, clmax, ldmax])
        
    # print("\nApplying moment filter...")
    # airfoils, properties = filter_cm_at_alpha(airfoils, properties, 3.0, -0.05, 0.05)
    
    print("\nFetching top 30 airfoils...")
    return filter_top_clmax(airfoils, properties)

# Allow Streamlit's native cache spinner to handle the loading state
@st.cache_data(show_spinner="Fetching geometry from UIUC...") 
def fetch_and_parse_airfoil_coords(airfoil_name):
    """Fetches and parses airfoil coordinates from the UIUC database."""
    url = f"https://m-selig.ae.illinois.edu/ads/coord/{airfoil_name.lower()}.dat"
    try:
        response = requests.get(url)
        response.raise_for_status() 
        
        lines = response.text.strip().splitlines()
        if not lines:
            return None, None
            
        line_idx = 0
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            if not line: 
                line_idx += 1
                break
            if re.match(r"^\s*[-+]?\d*\.?\d+\s+[-+]?\d*\.?\d+\s*$", line):
                break 
            line_idx += 1

        def parse_points_from_block(start_idx, end_idx):
            coords = []
            for i in range(start_idx, end_idx):
                line = lines[i].strip()
                if not line:
                    break 
                parts = line.split()
                if len(parts) == 2:
                    try:
                        coords.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        pass 
            return np.array(coords)

        gap_idx = -1
        for i in range(line_idx, len(lines)):
            if not lines[i].strip():
                gap_idx = i
                break
        
        if gap_idx == -1:
            return None, None

        upper_surface = parse_points_from_block(line_idx, gap_idx)
        lower_surface = parse_points_from_block(gap_idx + 1, len(lines))
        
        if len(upper_surface) == 0 or len(lower_surface) == 0:
             return None, None

        return upper_surface, lower_surface

    except Exception:
        return None, None

with st.spinner("Computing airfoil aerodynamics via NeuralFoil..."):
    all_airfoils, all_properties = load_and_compute_data()

# Setup consistent colors for the selected airfoils globally
colors = plt.cm.turbo(np.linspace(0, 1, len(all_airfoils))) 
color_map = {name: colors[i] for i, name in enumerate(all_airfoils.keys())}

# SIDEBAR: Controls & Legend
st.sidebar.header("Controls & Legend")

selected_airfoils = []
st.sidebar.markdown("### Airfoil Selection")

# Loop through all available airfoils
for af_name in all_airfoils.keys():
    # Create two columns in the sidebar: a small one for the color, a larger one for the checkbox
    col_color, col_check = st.sidebar.columns([1, 6])
    
    color_hex = mcolors.to_hex(color_map[af_name])
    
    with col_color:
        # Draw the colored square. 'margin-top' pushes it down slightly to vertically align with the checkbox text.
        st.markdown(f"<div style='width: 18px; height: 18px; background-color: {color_hex}; border-radius: 4px; margin-top: 8px; border: 1px solid rgba(255,255,255,0.2);'></div>", unsafe_allow_html=True)
        
    with col_check:
        # If the checkbox is checked, add it to our active list
        if st.checkbox(af_name.upper(), value=True, key=f"chk_{af_name}"):
            selected_airfoils.append(af_name)

# Filter data based on selection
filtered_airfoils = {k: v for k, v in all_airfoils.items() if k in selected_airfoils}
filtered_properties = [p for p in all_properties if p[0] in selected_airfoils]

# Main Layout: Split screen into two halves
left_half, right_half = st.columns(2, gap="large")

# LEFT HALF: Plots
with left_half:
    st.header("Aerodynamic Polars")
    
    def create_polar_plot(x_data, y_data, title, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(4, 3)) 
        for name, data in filtered_airfoils.items():
            alpha, cl, cd, cm = data
            ld = cl / cd
            
            x = locals()[x_data]
            y = locals()[y_data]
            
            ax.plot(x, y, label=name, color=color_map[name])
            
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.7)
        return fig

    if filtered_airfoils:
        polar_col1, polar_col2 = st.columns(2)
        
        with polar_col1:
            st.pyplot(create_polar_plot('alpha', 'cl', 'Cl vs Alpha', 'Alpha (deg)', 'Cl'))
            st.pyplot(create_polar_plot('alpha', 'cm', 'Cm vs Alpha', 'Alpha (deg)', 'Cm'))
            st.pyplot(create_polar_plot('cd', 'cl', 'Drag Polar (Cl vs Cd)', 'Cd', 'Cl'))
            
        with polar_col2:
            st.pyplot(create_polar_plot('alpha', 'cd', 'Cd vs Alpha', 'Alpha (deg)', 'Cd'))
            st.pyplot(create_polar_plot('alpha', 'ld', 'L/D vs Alpha', 'Alpha (deg)', 'L/D'))
    else:
        st.info("Please select at least one airfoil from the sidebar.")

    st.divider()

    st.header("Pareto Front: Lift vs. Efficiency")

    if filtered_properties:
        df = pd.DataFrame(filtered_properties, columns=['Name', 'Cl_max', 'LD_max'])
        df_sorted = df.sort_values(by=['LD_max', 'Cl_max'], ascending=[False, False]).reset_index(drop=True)
        
        pareto_front = []
        max_cl_seen = -np.inf
        for index, row in df_sorted.iterrows():
            if row['Cl_max'] > max_cl_seen:
                pareto_front.append(row)
                max_cl_seen = row['Cl_max']
                
        pareto_df = pd.DataFrame(pareto_front).sort_values(by='LD_max')
        
        fig_pareto, ax_pareto = plt.subplots(figsize=(8, 5))
        
        ax_pareto.scatter(df['LD_max'], df['Cl_max'], color='steelblue', alpha=0.5, s=50, label='Airfoils')
        ax_pareto.plot(pareto_df['LD_max'], pareto_df['Cl_max'], color='crimson', marker='o', 
                 linestyle='-', linewidth=2, markersize=8, label='Pareto Front')
        
        for index, row in df.iterrows():
            ax_pareto.annotate(row['Name'], (row['LD_max'], row['Cl_max']), 
                               xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax_pareto.set_xlabel('Maximum L/D Ratio (Efficiency)')
        ax_pareto.set_ylabel('Maximum Lift Coefficient (Cl_max)')
        ax_pareto.grid(True, linestyle='--', alpha=0.7)
        ax_pareto.legend()
        
        st.pyplot(fig_pareto)

# RIGHT HALF: Airfoil Geometry
with right_half:
    st.header("Airfoil Geometry Profiles")
    
    if not selected_airfoils:
        st.info("Select airfoils to view their physical geometry.")
    else:
        for af_name in selected_airfoils:
            upper, lower = fetch_and_parse_airfoil_coords(af_name)
            
            if upper is not None and lower is not None:
                fig_geo, ax_geo = plt.subplots(figsize=(6, 1.5))
                
                ax_geo.plot(upper[:, 0], upper[:, 1], color=color_map[af_name], linewidth=1.5)
                ax_geo.plot(lower[:, 0], lower[:, 1], color=color_map[af_name], linewidth=1.5)
                
                ax_geo.fill(
                    np.append(upper[:, 0], lower[::-1, 0]),
                    np.append(upper[:, 1], lower[::-1, 1]),
                    color=color_map[af_name], alpha=0.2
                )
                
                ax_geo.axis('equal') 
                
                ax_geo.set_title(f"{af_name.upper()}", fontsize=12, fontweight='bold', color=color_map[af_name])
                ax_geo.grid(True, linestyle=':', alpha=0.6)
                
                ax_geo.set_xticks([])
                ax_geo.set_yticks([])
                
                st.pyplot(fig_geo)
            else:
                st.warning(f"Could not retrieve coordinate data for '{af_name}' from the UIUC database.")