import pandas as pd
import numpy as np
import requests
import io
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
import time

def main():
    airfoils = {}
    properties = []
    reynolds_number = 100000
    
    for airfoil in search_airfoils_by_geometry(5, 15, 0, 10):
        (alphas, cl, cd, cm), (clmax, ldmax) = fetch_af_polar(airfoil, reynolds_number)
        airfoils[airfoil] = alphas, cl , cd, cm
        properties.append([airfoil, clmax, ldmax])
        
    print("Plotting airfoils. ----->")
    plot_polars(airfoils)
    plot_pareto_frontier(properties)

def fetch_af_polar(af_name, re_num):
    url = f"http://airfoiltools.com/polar/csv?polar=xf-{af_name}-il-{re_num}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

    try:
        print(f"  -> FETCHING: {af_name} at Re={re_num} from AirfoilTools...")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), skiprows=10)
        df.columns = df.columns.str.strip()        
    except requests.exceptions.RequestException as e:
        print(f"  -> ERROR: Failed to fetch {af_name}: {e}")
        return None, None, None, None, None

    alpha = df['Alpha'].to_numpy()
    cl = df['Cl'].to_numpy()
    cd = df['Cd'].to_numpy()
    cm = df['Cm'].to_numpy()
    
    ld = cl / cd
    cl_max = np.max(cl)
    ld_max = np.max(ld)
    alpha_stall = alpha[np.argmax(cl)]
    alpha_ld_max = alpha[np.argmax(ld)]
    
    return (alpha, cl, cd, cm), [cl_max, ld_max]

def search_airfoils_by_geometry(min_thick=2.0, max_thick=66.4, min_camber=0.0, max_camber=16.4, max_results=200):
    url = "http://airfoiltools.com/search/index"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    airfoil_names = []
    page_num = 1
    print(f"Searching database for Thickness: {min_thick}-{max_thick}% | Camber: {min_camber}-{max_camber}%...")
    print(f"Targeting up to {max_results} airfoils.\n")
    
    try:
        while len(airfoil_names) < max_results:
            payload = {
                "m[textSearch]": "",
                "m[maxThickness]": max_thick,
                "m[minThickness]": min_thick,
                "m[maxCamber]": max_camber,
                "m[minCamber]": min_camber,
                "m[grp]": "",
                "m[sort]": 9,
                "m[page]": page_num # Increment this variable to turn the page
            }
            
            print(f"  -> Scraping page {page_num}...")
            response = requests.get(url, params=payload, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=re.compile(r'airfoil='))
            
            added_on_page = 0
            for link in links:
                if len(airfoil_names) >= max_results:
                    return airfoil_names 
                    
                href = link.get('href')
                match = re.search(r'airfoil=([a-zA-Z0-9_\-]+)', href)                
                if match:
                    clean_name = match.group(1).replace("-il", "")
                    if clean_name not in airfoil_names:
                        airfoil_names.append(clean_name)
                        added_on_page += 1
            
            if added_on_page == 0:
                return airfoil_names
            page_num += 1
            time.sleep(1) 

    except requests.exceptions.RequestException as e:
        print(f"  -> [ERROR] Failed to search database: {e}")
        return []

def plot_polars(airfoil_data_dict):
    # Set up a 2x3 grid of plots
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Airfoil Aerodynamic Polars', fontsize=20, fontweight='bold')
    
    for name, data in airfoil_data_dict.items():
        alpha, cl, cd, cm = data
        ld = cl / cd 
        
        # Cl vs Alpha
        axs[0, 0].plot(alpha, cl, label=name)
        axs[0, 0].set_title('Cl vs Alpha', fontsize=14)
        axs[0, 0].set_xlabel('Alpha (deg)')
        axs[0, 0].set_ylabel('Cl')
        axs[0, 0].grid(True)
        
        # Cd vs Alpha
        axs[0, 1].plot(alpha, cd, label=name)
        axs[0, 1].set_title('Cd vs Alpha', fontsize=14)
        axs[0, 1].set_xlabel('Alpha (deg)')
        axs[0, 1].set_ylabel('Cd')
        axs[0, 1].grid(True)
        
        # Cm vs Alpha
        axs[0, 2].plot(alpha, cm, label=name)
        axs[0, 2].set_title('Cm vs Alpha', fontsize=14)
        axs[0, 2].set_xlabel('Alpha (deg)')
        axs[0, 2].set_ylabel('Cm')
        axs[0, 2].grid(True)
        
        # L/D vs Alpha
        axs[1, 0].plot(alpha, ld, label=name)
        axs[1, 0].set_title('L/D vs Alpha', fontsize=14)
        axs[1, 0].set_xlabel('Alpha (deg)')
        axs[1, 0].set_ylabel('L/D')
        axs[1, 0].grid(True)
        
        # Cl vs Cd (Drag Polar)
        axs[1, 1].plot(cd, cl, label=name)
        axs[1, 1].set_title('Cl vs Cd (Drag Polar)', fontsize=14)
        axs[1, 1].set_xlabel('Cd')
        axs[1, 1].set_ylabel('Cl')
        axs[1, 1].grid(True)
    
    axs[1, 2].axis('off')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[1, 2].legend(handles, labels, loc='center', fontsize=11, ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.93], h_pad=3.0, w_pad=2.0)
    plt.show()
    
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

if __name__ == "__main__":
    main()