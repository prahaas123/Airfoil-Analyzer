import pandas as pd
import numpy as np
import os
import json
import neuralfoil as nf
import aerosandbox as asb
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

def main():
    airfoils = {}
    properties = []
    reynolds_number = 100000
    
    for airfoil in search_airfoils_by_geometry(10, 15, 0, 10):
        (alphas, cl, cd, cm), (clmax, ldmax) = fetch_af_polar(airfoil, reynolds_number)
        if alphas is not None:
            airfoils[airfoil] = alphas, cl , cd, cm
            properties.append([airfoil, clmax, ldmax])
        
    print("\nApplying moment filter...")
    airfoils, properties = filter_cm_at_alpha(airfoils, properties, 3.0, -0.05, 0.05)
    
    print("\nFetching top 30 airfoils...")
    airfoils, properties = filter_top_clmax(airfoils, properties)
        
    print("Plotting airfoils. ----->")
    plot_polars(airfoils)
    plot_pareto_frontier(properties)

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

def plot_polars(airfoil_data_dict):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Airfoil Aerodynamic Polars', fontsize=20, fontweight='bold')
    
    lines_by_airfoil = {name: [] for name in airfoil_data_dict.keys()}
    airfoil_colors = {}
    
    for name, data in airfoil_data_dict.items():
        alpha, cl, cd, cm = data
        ld = cl / cd 
        
        line, = axs[0, 0].plot(alpha, cl, label=name)
        lines_by_airfoil[name].append(line)
        airfoil_colors[name] = line.get_color() 
        
        axs[0, 0].set_title('Cl vs Alpha', fontsize=14)
        axs[0, 0].set_xlabel('Alpha (deg)')
        axs[0, 0].set_ylabel('Cl')
        axs[0, 0].grid(True)
        
        # Apply the exact same color to the rest of the graphs to ensure consistency
        line, = axs[0, 1].plot(alpha, cd, label=name, color=airfoil_colors[name])
        lines_by_airfoil[name].append(line)
        axs[0, 1].set_title('Cd vs Alpha', fontsize=14)
        axs[0, 1].set_xlabel('Alpha (deg)')
        axs[0, 1].set_ylabel('Cd')
        axs[0, 1].grid(True)
        
        line, = axs[0, 2].plot(alpha, cm, label=name, color=airfoil_colors[name])
        lines_by_airfoil[name].append(line)
        axs[0, 2].set_title('Cm vs Alpha', fontsize=14)
        axs[0, 2].set_xlabel('Alpha (deg)')
        axs[0, 2].set_ylabel('Cm')
        axs[0, 2].grid(True)
        
        line, = axs[1, 0].plot(alpha, ld, label=name, color=airfoil_colors[name])
        lines_by_airfoil[name].append(line)
        axs[1, 0].set_title('L/D vs Alpha', fontsize=14)
        axs[1, 0].set_xlabel('Alpha (deg)')
        axs[1, 0].set_ylabel('L/D')
        axs[1, 0].grid(True)
        
        line, = axs[1, 1].plot(cd, cl, label=name, color=airfoil_colors[name])
        lines_by_airfoil[name].append(line)
        axs[1, 1].set_title('Cl vs Cd (Drag Polar)', fontsize=14)
        axs[1, 1].set_xlabel('Cd')
        axs[1, 1].set_ylabel('Cl')
        axs[1, 1].grid(True)
    
    axs[1, 2].axis('off')
    plt.subplots_adjust(left=0.05, right=0.80, top=0.90, bottom=0.10, wspace=0.3, hspace=0.3)
    rax = fig.add_axes([0.82, 0.05, 0.16, 0.85]) 
    
    labels = list(airfoil_data_dict.keys())
    visibility = [True] * len(labels)
    check = CheckButtons(rax, labels, visibility)

    for i, label in enumerate(check.labels):
        label.set_fontsize(8)
        label.set_color(airfoil_colors[labels[i]])
        label.set_fontweight('bold')

    def toggle_lines(label):
        for line in lines_by_airfoil[label]:
            line.set_visible(not line.get_visible())
        fig.canvas.draw_idle()

    check.on_clicked(toggle_lines)
    plt.show()
    
    return check
    
def plot_pareto_frontier(airfoil_properties):
    df = pd.DataFrame(airfoil_properties, columns=['Name', 'Cl_max', 'LD_max'])
    df_sorted = df.sort_values(by=['LD_max', 'Cl_max'], ascending=[False, False]).reset_index(drop=True)
    pareto_front = []
    max_cl_seen = -np.inf
    
    for index, row in df_sorted.iterrows():
        if row['Cl_max'] > max_cl_seen:
            pareto_front.append(row)
            max_cl_seen = row['Cl_max']
            
    pareto_df = pd.DataFrame(pareto_front)
    plt.figure(figsize=(12, 8))
    plt.scatter(df['LD_max'], df['Cl_max'], color='steelblue', alpha=0.5, s=50, label='Airfoils')
    pareto_df = pareto_df.sort_values(by='LD_max') 
    plt.plot(pareto_df['LD_max'], pareto_df['Cl_max'], color='crimson', marker='o', 
             linestyle='-', linewidth=2, markersize=8, label='Pareto Front')
    
    for index, row in df.iterrows():
        if row['Name'] in pareto_df['Name'].values:
            plt.annotate(
                row['Name'], 
                (row['LD_max'], row['Cl_max']), 
                xytext=(8, 5),
                textcoords='offset points', 
                fontsize=11, 
                fontweight='bold',
                color='darkred'
            )
        else:
            plt.annotate(
                row['Name'], 
                (row['LD_max'], row['Cl_max']), 
                xytext=(5, 5),
                textcoords='offset points', 
                fontsize=6, 
                alpha=0.6, 
                color='black'
            )

    plt.title('Airfoil Pareto Front: Lift vs. Efficiency', fontsize=18, fontweight='bold')
    plt.xlabel('Maximum L/D Ratio (Efficiency)', fontsize=14)
    plt.ylabel('Maximum Lift Coefficient ($C_{l,max}$)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='lower right')
    plt.show()
    
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

def filter_top_clmax(airfoils, properties, top_n= 30):
    sorted_properties = sorted(properties, key=lambda x: x[1], reverse=True)
    top_properties = sorted_properties[:top_n]
    top_names = {prop[0] for prop in top_properties}
    filtered_airfoils = {name: data for name, data in airfoils.items() if name in top_names}
    print(f"\n  -> Filtered down to the top {len(filtered_airfoils)} airfoils by Cl_max.")
    return filtered_airfoils, top_properties

if __name__ == "__main__":
    main()