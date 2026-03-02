import pandas as pd
import numpy as np
import requests
import io
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re

def main():
    airfoils = {}
    reynolds_number = 100000
    
    for airfoil in search_airfoils_by_geometry(10, 15, 10):
        airfoils[airfoil] = fetch_af_polar(airfoil, reynolds_number)
        
    plot_polars(airfoils)

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
        return None

    alpha = df['Alpha'].to_numpy()
    cl = df['Cl'].to_numpy()
    cd = df['Cd'].to_numpy()
    cm = df['Cm'].to_numpy()
    
    return alpha, cl, cd, cm

def search_airfoils_by_geometry(min_thick=2.0, max_thick=66.4, min_camber=0.0, max_camber=16.4):
    url = "http://airfoiltools.com/search/index"
    payload = {
        "MAirfoilSearchForm[textSearch]": "",
        "MAirfoilSearchForm[maxThickness]": max_thick,
        "MAirfoilSearchForm[minThickness]": min_thick,
        "MAirfoilSearchForm[maxCamber]": max_camber,
        "MAirfoilSearchForm[minCamber]": min_camber,
        "MAirfoilSearchForm[grp]": "",
        "MAirfoilSearchForm[sort]": 9, # Sorts by Max Cl/Cd at Re=50,000 (Perfect for Micro Class)
        "yt0": "Search" 
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    print(f"Searching database for Thickness: {min_thick}-{max_thick}% | Camber: {min_camber}-{max_camber}%...")
    
    try:
        response = requests.get(url, params=payload, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=re.compile(r'airfoil='))
        
        airfoil_names = []
        for link in links:
            href = link.get('href')
            match = re.search(r'airfoil=([a-zA-Z0-9_\-]+)', href)
            
            if match:
                raw_name = match.group(1)
                clean_name = raw_name.replace("-il", "")
                if clean_name not in airfoil_names:
                    airfoil_names.append(clean_name)
                    
        print(f"  -> Found {len(airfoil_names)} matching airfoils.")
        return airfoil_names

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

if __name__ == "__main__":
    main()