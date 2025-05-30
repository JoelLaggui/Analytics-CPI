import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import numpy as np
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Philippines CPI Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Design constants
PRIMARY_COLOR = "#2563eb"
BG_COLOR = "#f8fafc"
CARD_COLOR = "#ffffff"
ACCENT_COLOR = "#fbbf24"
TEXT_COLOR = "#1e293b"
GOOD_COLOR = "#22c55e"  # Green for low CPI (good)
BAD_COLOR = "#ef4444"   # Red for high CPI (bad)

# Add custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }
    .css-1d391kg {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 12px rgba(37,99,235,0.08);
    }
    h1, h2, h3 {
        color: #2563eb;
    }
    .kpi-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        box-shadow: 0 2px 8px rgba(30,41,59,0.08);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(37,99,235,0.15);
    }
    .rank-meter {
        height: 30px;
        background: linear-gradient(90deg, rgba(34,197,94,0.3) 0%, rgba(251,191,36,0.3) 50%, rgba(239,68,68,0.3) 100%);
        border-radius: 15px;
        position: relative;
        margin: 20px 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-card {
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(37,99,235,0.08);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load the data
@st.cache_data
def load_data():
    # Try to load the data from a relative path first
    data_path = 'cleaned_Data.csv'
    if not os.path.exists(data_path):
        # Fall back to the absolute path if the file doesn't exist in the current directory
        data_path = r"C:\Users\jenal\ForDataAnalysis\cleaned_Data.csv"
    df = pd.read_csv(data_path)
    return df

@st.cache_data
def load_geojson():
    # Try to load the geojson from a relative path first
    geojson_path = "Regions.json"
    if not os.path.exists(geojson_path):
        # Fall back to the absolute path if the file doesn't exist in the current directory
        geojson_path = r"C:\Users\jenal\ForDataAnalysis\Regions.json"
    with open(geojson_path) as f:
        geojson_data = json.load(f)
    return geojson_data

# Load data
df = load_data()
geojson_data = load_geojson()

# Create a region mapping for GeoJSON compatibility
region_map = {
    "National Capital Region (NCR)": "Metropolitan Manila",
    "Cordillera Administrative Region (CAR)": "Cordillera Administrative Region (CAR)",
    "Region I (Ilocos Region)": "Ilocos Region (Region I)",
    "Region II (Cagayan Valley)": "Cagayan Valley (Region II)",
    "Region III (Central Luzon)": "Central Luzon (Region III)",
    "Region IV-A (CALABARZON)": "CALABARZON (Region IV-A)",
    "Region IV-B (MIMAROPA)": "MIMAROPA (Region IV-B)",
    "Region V (Bicol Region)": "Bicol Region (Region V)",
    "Region VI (Western Visayas)": "Western Visayas (Region VI)",
    "Region VII (Central Visayas)": "Central Visayas (Region VII)",
    "Region VIII (Eastern Visayas)": "Eastern Visayas (Region VIII)",
    "Region IX (Zamboanga Peninsula)": "Zamboanga Peninsula (Region IX)",
    "Region X (Northern Mindanao)": "Northern Mindanao (Region X)",
    "Region XI (Davao Region)": "Davao Region (Region XI)",
    "Region XII (SOCCSKSARGEN)": "SOCCSKSARGEN (Region XII)",
    "Region XIII (Caraga)": "Caraga (Region XIII)",
    "Autonomous Region in Muslim Mindanao (ARMM)": "Autonomous Region of Muslim Mindanao (ARMM)"
}

# Add mapping columns to the dataframe
df["Region_Match"] = df["Geolocation"].map(region_map)

# Add short region codes
region_short = {
    "Metropolitan Manila": "NCR",
    "Cordillera Administrative Region (CAR)": "CAR",
    "Ilocos Region (Region I)": "REGION I",
    "Cagayan Valley (Region II)": "REGION II",
    "Central Luzon (Region III)": "REGION III",
    "CALABARZON (Region IV-A)": "REGION IV-A",
    "MIMAROPA (Region IV-B)": "REGION IV-B",
    "Bicol Region (Region V)": "REGION V",
    "Western Visayas (Region VI)": "REGION VI",
    "Central Visayas (Region VII)": "REGION VII",
    "Eastern Visayas (Region VIII)": "REGION VIII",
    "Zamboanga Peninsula (Region IX)": "REGION IX",
    "Northern Mindanao (Region X)": "REGION X",
    "Davao Region (Region XI)": "REGION XI",
    "SOCCSKSARGEN (Region XII)": "REGION XII",
    "Caraga (Region XIII)": "REGION XIII",
    "Autonomous Region of Muslim Mindanao (ARMM)": "ARMM"
}
df["REGION"] = df["Region_Match"].map(region_short)

# App title and header
st.title("Analysis Of Consumer Price Index Trends in the Philippines (1994-2018)")
st.subheader("Interactive visualization and analysis dashboard")
st.markdown("<p style='text-align: center; color: #fbbf24; font-weight: bold;'>Team Members: Joel L. Laggui Jr. & Fernando C. Mansibang</p>", unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["Dashboard", "Key Insights & Analysis"])

# Initialize session state for selected region
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = None

# Sidebar filters
with st.sidebar:
    st.header("Dashboard Controls")
    
    selected_year = st.selectbox(
        "Select Year:",
        options=sorted(df["Year"].unique()),
        index=len(df["Year"].unique())-1  # Default to most recent year
    )
    
    selected_commodity = st.selectbox(
        "Select Commodity:",
        options=sorted(df["Commodity"].unique()),
        index=0
    )
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #fbbf24; font-weight: bold;'>Tip: Click a region on the map to see detailed information. Click 'Reset Selection' to unselect.</p>", unsafe_allow_html=True)
    
    if st.button("Reset Region Selection"):
        st.session_state.selected_region = None
        st.rerun()

# Create KPI cards
def create_kpi_cards(selected_year, selected_commodity, selected_region):
    filtered = df[(df["Year"] == selected_year) & 
                 (df["Commodity"] == selected_commodity) & 
                 (df["Region_Match"].notnull())]
    
    prev_year = selected_year - 1
    prev_filtered = df[(df["Year"] == prev_year) & (df["Commodity"] == selected_commodity) & (df["Region_Match"].notnull())]
    prev_total = prev_filtered['CPI'].sum() if not prev_filtered.empty else None
    prev_avg = prev_filtered['CPI'].mean() if not prev_filtered.empty else None
    
    if selected_region and selected_region in filtered['Geolocation'].values:
        filtered_region = filtered[filtered['Geolocation'] == selected_region]
        region_label = f"for {selected_region}"
        # Compute rank among all regions for this year/commodity
        all_regions = filtered.copy()
        region_cpi = filtered_region['CPI'].iloc[0]
        # Lower CPI is better, so rank is based on ascending order
        rank = (all_regions['CPI'] < region_cpi).sum() + 1
        total_regions = all_regions.shape[0]
        rank_str = f" (Rank: {rank}/{total_regions})"
    else:
        filtered_region = filtered
        region_label = "(All Regions)"
        rank_str = ""
    
    # Create columns for KPI cards
    col1, col2, col3, col4 = st.columns(4)
    
    # Total CPI
    total_cpi = filtered_region['CPI'].sum()
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <h2 style="color: {PRIMARY_COLOR}; font-weight: bold; font-size: 2.1rem; margin: 0;">{total_cpi:.2f}</h2>
            <p style="color: {TEXT_COLOR}; font-size: 1.1rem; margin: 0;">
                Total CPI {region_label}{rank_str}
                {f'<span style="color: {BAD_COLOR if (prev_total and (total_cpi - prev_total) / prev_total * 100 > 0) else GOOD_COLOR}">({(total_cpi - prev_total) / prev_total * 100:+.2f}%)</span>' if prev_total and prev_total != 0 else ''}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Average CPI
    avg_cpi = filtered_region['CPI'].mean()
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <h2 style="color: {ACCENT_COLOR}; font-weight: bold; font-size: 2.1rem; margin: 0;">{avg_cpi:.2f}</h2>
            <p style="color: {TEXT_COLOR}; font-size: 1.1rem; margin: 0;">
                Average CPI
                {f'<span style="color: {BAD_COLOR if (prev_avg and (avg_cpi - prev_avg) / prev_avg * 100 > 0) else GOOD_COLOR}">({(avg_cpi - prev_avg) / prev_avg * 100:+.2f}%)</span>' if prev_avg and prev_avg != 0 else ''}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Min CPI (Best) - Lowest CPI is best
    if not filtered.empty:
        min_idx = filtered['CPI'].idxmin()
        min_loc = filtered.loc[min_idx, 'Geolocation']
        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <h2 style="color: {GOOD_COLOR}; font-weight: bold; font-size: 2.1rem; margin: 0;">{filtered['CPI'].min():.2f}</h2>
                <p style="color: {TEXT_COLOR}; font-size: 1.1rem; margin: 0;">
                    Best CPI: <span style="font-weight: bold;">{min_loc}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Max CPI (Worst) - Highest CPI is worst
    if not filtered.empty:
        max_idx = filtered['CPI'].idxmax()
        max_loc = filtered.loc[max_idx, 'Geolocation']
        with col4:
            st.markdown(f"""
            <div class="kpi-card">
                <h2 style="color: {BAD_COLOR}; font-weight: bold; font-size: 2.1rem; margin: 0;">{filtered['CPI'].max():.2f}</h2>
                <p style="color: {TEXT_COLOR}; font-size: 1.1rem; margin: 0;">
                    Worst CPI: <span style="font-weight: bold;">{max_loc}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

# Create the choropleth map
def create_map(selected_year, selected_commodity, selected_region):
    filtered = df[(df["Year"] == selected_year) & 
                 (df["Commodity"] == selected_commodity) & 
                 (df["Region_Match"].notnull())]
    
    # Compute national average for this year/commodity
    national_avg = filtered['CPI'].mean()
    
    # Compute previous year % change
    prev_year = selected_year - 1
    prev_filtered = df[(df["Year"] == prev_year) & (df["Commodity"] == selected_commodity) & (df["Region_Match"].notnull())]
    prev_cpi_map = prev_filtered.set_index('Geolocation')['CPI'].to_dict()
    
    # Compute rank and % change for each region
    filtered = filtered.copy()
    # Lower CPI is better, so rank is based on ascending order
    filtered['Rank'] = filtered['CPI'].rank(ascending=True, method='min').astype(int)
    filtered['AboveAvg'] = filtered['CPI'] > national_avg
    
    def pct_change(row):
        prev = prev_cpi_map.get(row['Geolocation'])
        if prev is not None and prev != 0:
            return ((row['CPI'] - prev) / prev) * 100
        return None
    
    filtered['%Change'] = filtered.apply(pct_change, axis=1)
    
    # Dynamic color scale
    cmin = filtered['CPI'].min()
    cmax = filtered['CPI'].max()
    
    # Blur/fade effect: set opacity lower for unselected regions
    if selected_region and selected_region in filtered['Geolocation'].values:
        filtered['opacity'] = filtered['Geolocation'].apply(lambda x: 1.0 if x == selected_region else 0.25)
    else:
        filtered['opacity'] = 0.85
    
    # Custom hover text
    def hover_text(row):
        pct = f"{row['%Change']:+.2f}%" if row['%Change'] is not None else "N/A"
        above = "Above" if row['AboveAvg'] else "Below"
        # Above average is BAD (red), below average is GOOD (green)
        color = BAD_COLOR if row['AboveAvg'] else GOOD_COLOR
        return f"<b>{row['Geolocation']}</b><br>CPI: <b>{row['CPI']:.2f}</b><br>Rank: <b>{row['Rank']}</b><br>% Change: <b>{pct}</b><br><span style='color:{color}'>{above} National Avg</span>"
    
    filtered['hover'] = filtered.apply(hover_text, axis=1)
    
    # Map - Use a color scale where red is high CPI (bad) and green is low CPI (good)
    fig = px.choropleth_mapbox(
        filtered,
        geojson=geojson_data,
        locations='Region_Match',
        featureidkey="properties.REGION",
        color='CPI',
        color_continuous_scale=["#22c55e", "#fbbf24", "#ef4444"],  # Green to Yellow to Red
        mapbox_style="open-street-map",
        zoom=4.3,
        center={"lat": 12.8797, "lon": 121.7740},
        opacity=0.85,
        height=500,
        hover_name=None,
        custom_data=['Geolocation', 'CPI', 'Rank', '%Change', 'AboveAvg', 'hover'],
        labels={'CPI': 'CPI'}
    )
    
    # Enable zoom and pan controls
    fig.update_layout(
        mapbox=dict(
            accesstoken=None,  # Use default public token
            zoom=4.3,
            center={"lat": 12.8797, "lon": 121.7740},
            style="open-street-map"
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        font=dict(family="Segoe UI, Arial", color=TEXT_COLOR),
        title=dict(
            text=f"CPI by Region for {selected_commodity} in {selected_year}",
            font=dict(size=18, color=PRIMARY_COLOR),
            x=0.5,
            xanchor='center'
        ),
        hoverlabel=dict(
            bgcolor=PRIMARY_COLOR,
            font_size=14,
            font_family="Segoe UI, Arial"
        )
    )
    
    # Set color range
    fig.update_traces(zmin=cmin, zmax=cmax)
    
    # Custom hovertemplate
    fig.update_traces(hovertemplate="%{customdata[5]}")
    
    # Highlight selected region border and fade others
    if selected_region and selected_region in filtered['Geolocation'].values:
        # Fade all regions except selected
        opacities = filtered['Geolocation'].apply(lambda x: 1.0 if x == selected_region else 0.25).tolist()
        fig.update_traces(marker=dict(opacity=opacities))
        
        # Add a scattermapbox marker for selected region centroid
        sel = filtered[filtered['Geolocation'] == selected_region]
        region_match = sel['Region_Match'].iloc[0]
        for feature in geojson_data['features']:
            if feature['properties']['REGION'] == region_match:
                coords = feature['geometry']['coordinates']
                if feature['geometry']['type'] == 'MultiPolygon':
                    all_coords = [pt for poly in coords for ring in poly for pt in ring]
                else:
                    all_coords = [pt for ring in coords for pt in ring]
                lons = [pt[0] for pt in all_coords]
                lats = [pt[1] for pt in all_coords]
                centroid = {"lat": sum(lats)/len(lats), "lon": sum(lons)/len(lons)}
                fig.add_scattermapbox(
                    lat=[centroid['lat']], lon=[centroid['lon']],
                    mode='markers',
                    marker=dict(size=18, color='rgba(37,99,235,0.7)', symbol='star'),
                    showlegend=False, hoverinfo='skip'
                )
                break
    
    # Add annotation for national average
    fig.add_annotation(
        text=f"National Avg: {national_avg:.2f}",
        x=1.01, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=15, color=PRIMARY_COLOR),
        bgcolor="rgba(255,255,255,0.8)", bordercolor=PRIMARY_COLOR, borderwidth=1
    )
    
    # Add click events
    fig.update_layout(clickmode='event+select')
    
    return fig, filtered

# Create radar chart for commodity comparison
def create_radar_chart(selected_year, selected_commodity, selected_region):
    if not selected_region:
        # Show a message or empty chart if no region is selected
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text="Select a region on the map to compare commodities",
                font=dict(size=16, color=PRIMARY_COLOR),
                x=0.5,
                xanchor='center'
            ),
            paper_bgcolor=CARD_COLOR,
            plot_bgcolor=CARD_COLOR,
            font=dict(family="Segoe UI, Arial", color=TEXT_COLOR),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        return fig
    
    # Get data for selected region and year
    region_data = df[(df["Year"] == selected_year) & (df["Geolocation"] == selected_region)]
    
    fig = px.line_polar(
        region_data,
        r='CPI',
        theta='Commodity',
        line_close=True,
        color_discrete_sequence=[PRIMARY_COLOR],
        title=f"Commodity CPI Comparison for {selected_region} ({selected_year})"
    )
    
    fig.update_traces(fill='toself', line=dict(width=4))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                showticklabels=True,
                gridcolor='#e5e7eb',
                gridwidth=1,
                linecolor=PRIMARY_COLOR,
                linewidth=2
            ),
            angularaxis=dict(
                tickfont=dict(size=13, color=TEXT_COLOR),
                gridcolor='#e5e7eb',
                gridwidth=1
            )
        ),
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        font=dict(family="Segoe UI, Arial", color=TEXT_COLOR),
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(
            font=dict(size=20, color=PRIMARY_COLOR),
            x=0.5,
            xanchor='center'
        ),
        hovermode='closest',
        hoverlabel=dict(
            bgcolor=PRIMARY_COLOR,
            font_size=14,
            font_family="Segoe UI, Arial"
        )
    )
    
    return fig

# Create trend graph
def create_trend_graph(selected_commodity, selected_region):
    filtered = df[(df["Commodity"] == selected_commodity) & (df["Region_Match"].notnull())]
    
    # Compute national average per year
    national_avg = filtered.groupby('Year')['CPI'].mean().reset_index()
    
    # Bar chart trend: x=Year, y=CPI
    if selected_region and selected_region in filtered['Geolocation'].values:
        region_data = filtered[filtered['Geolocation'] == selected_region]
        title = f"CPI Trend for {selected_region} - {selected_commodity}"
        showlegend = False
        
        fig = px.bar(
            region_data,
            x="Year", y="CPI",
            color_discrete_sequence=[PRIMARY_COLOR],
            title=title,
            height=400,
            labels={"CPI": "CPI", "Year": "Year"}
        )
        
        # Add national average line
        fig.add_scatter(
            x=national_avg['Year'], y=national_avg['CPI'],
            mode='lines+markers',
            name='National Avg',
            line=dict(color=ACCENT_COLOR, width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond', color=ACCENT_COLOR),
            showlegend=True
        )
    else:
        # Show all regions as grouped bars
        region_data = filtered.copy()
        title = f"CPI Trend for All Regions - {selected_commodity}"
        showlegend = True
        
        fig = px.bar(
            region_data,
            x="Year", y="CPI", color="REGION",
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Safe,
            title=title,
            height=400,
            labels={"CPI": "CPI", "Year": "Year", "REGION": "Region"}
        )
        
        # Add national average line
        fig.add_scatter(
            x=national_avg['Year'], y=national_avg['CPI'],
            mode='lines+markers',
            name='National Avg',
            line=dict(color=ACCENT_COLOR, width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond', color=ACCENT_COLOR),
            showlegend=True
        )
    
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        font=dict(family="Segoe UI, Arial", color=TEXT_COLOR),
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=1.02,
            font=dict(size=13),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=PRIMARY_COLOR,
            borderwidth=1
        ),
        title=dict(
            font=dict(size=22, color=PRIMARY_COLOR),
            x=0.5,
            xanchor='center'
        ),
        hovermode='x unified',
        showlegend=showlegend
    )
    
    fig.update_traces(marker_line_width=2, marker_line_color=PRIMARY_COLOR)
    
    return fig

# Create rank visualization
def create_rank_visualization(selected_year, selected_commodity, selected_region):
    if not selected_region:
        return None
    
    # Get data for all regions for this year/commodity
    all_regions = df[(df["Year"] == selected_year) & 
                     (df["Commodity"] == selected_commodity) & 
                     (df["Region_Match"].notnull())]
    
    # Sort regions by CPI - Lower CPI is better (ascending=True)
    sorted_regions = all_regions.sort_values('CPI', ascending=True).reset_index(drop=True)
    
    # Find the selected region's rank
    selected_idx = sorted_regions[sorted_regions['Geolocation'] == selected_region].index[0]
    total_regions = len(sorted_regions)
    rank = selected_idx + 1
    
    # Determine rank quality (lower rank = better)
    if rank <= total_regions / 3:
        rank_quality = "Good"
        rank_color = GOOD_COLOR
    elif rank <= 2 * total_regions / 3:
        rank_quality = "Average"
        rank_color = ACCENT_COLOR
    else:
        rank_quality = "High"
        rank_color = BAD_COLOR
    
    st.markdown(f"<h3 style='text-align: center; color: {PRIMARY_COLOR}; font-weight: bold; margin-bottom: 15px;'>Price Index Ranking for {selected_region}</h3>", unsafe_allow_html=True)
    
    # Rank position indicator with explanation
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center;">
            <h2 style="color: {rank_color}; font-weight: bold; font-size: 2rem; margin: 0 0 5px 0;">Rank: {rank}/{total_regions}</h2>
            <p style="color: {rank_color}; font-weight: bold; font-size: 1.2rem; margin: 0 0 15px 0;">Price Level: {rank_quality}</p>
            <p style="color: {TEXT_COLOR}; font-size: 1rem; font-style: italic; margin: 0 0 15px 0;">Lower rank = Lower prices = Better for consumers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Visual rank bar - Left is now best (lowest CPI)
        st.markdown(f"""
        <div class="rank-meter">
            <div style="position: absolute; left: {(rank-1)/(total_regions-1)*100 if total_regions > 1 else 50}%; top: -15px; transform: translateX(-50%); width: 30px; height: 30px; border-radius: 50%; background-color: {rank_color}; border: 4px solid white; box-shadow: 0 2px 8px rgba(37,99,235,0.3); z-index: 2;"></div>
            <div style="position: absolute; left: 0%; top: 35px; font-size: 1rem; font-weight: bold; color: {GOOD_COLOR};">Best<br>(Lowest Prices)</div>
            <div style="position: absolute; right: 0%; top: 35px; font-size: 1rem; font-weight: bold; color: {BAD_COLOR};">Worst<br>(Highest Prices)</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Region comparison - Lowest CPI is now best
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: rgba(34,197,94,0.1); border-radius: 8px; padding: 10px; text-align: center;">
            <h4 style="margin: 0; color: {GOOD_COLOR}; font-size: 1.1rem;">Best Region</h4>
            <p style="margin: 0; font-weight: bold; font-size: 1.05rem;">{sorted_regions.iloc[0]['Geolocation']}</p>
            <p style="margin: 0; font-size: 1rem;">CPI: {sorted_regions.iloc[0]['CPI']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="border: 2px solid {PRIMARY_COLOR}; border-radius: 8px; padding: 10px; text-align: center; box-shadow: 0 4px 6px rgba(37,99,235,0.1);">
            <h4 style="margin: 0; color: {PRIMARY_COLOR}; font-size: 1.1rem;">Selected Region</h4>
            <p style="margin: 0; font-weight: bold; font-size: 1.05rem;">{selected_region}</p>
            <p style="margin: 0; font-size: 1rem;">CPI: {sorted_regions.iloc[selected_idx]['CPI']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background-color: rgba(239,68,68,0.1); border-radius: 8px; padding: 10px; text-align: center;">
            <h4 style="margin: 0; color: {BAD_COLOR}; font-size: 1.1rem;">Worst Region</h4>
            <p style="margin: 0; font-weight: bold; font-size: 1.05rem;">{sorted_regions.iloc[-1]['Geolocation']}</p>
            <p style="margin: 0; font-size: 1rem;">CPI: {sorted_regions.iloc[-1]['CPI']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

# Generate insights
def generate_insights(selected_year, selected_commodity, selected_region):
    # Get filtered data
    filtered = df[(df["Year"] == selected_year) & 
                 (df["Commodity"] == selected_commodity) & 
                 (df["Region_Match"].notnull())]
    
    # Get previous year data for comparison
    prev_year = selected_year - 1
    prev_filtered = df[(df["Year"] == prev_year) & 
                      (df["Commodity"] == selected_commodity) & 
                      (df["Region_Match"].notnull())]
    
    # Get historical data for trend analysis
    historical = df[(df["Commodity"] == selected_commodity) & 
                   (df["Region_Match"].notnull())]
    
    # Create insights header
    st.markdown("<h2 style='color: #2563eb; font-weight: bold; font-size: 1.8rem; margin-bottom: 20px; text-align: center;'>📊 Key Insights & Analysis</h2>", unsafe_allow_html=True)
    
    # Insight 1: National Overview
    national_avg = filtered['CPI'].mean()
    national_std = filtered['CPI'].std()
    regions_above_avg = (filtered['CPI'] > national_avg).sum()
    total_regions = len(filtered)
    
    if not prev_filtered.empty:
        prev_national_avg = prev_filtered['CPI'].mean()
        national_change = ((national_avg - prev_national_avg) / prev_national_avg) * 100
        change_color = BAD_COLOR if national_change > 0 else GOOD_COLOR
        change_text = f"{national_change:+.1f}%" 
    else:
        change_text = "N/A"
        change_color = TEXT_COLOR
    
    st.markdown(f"""
    <div class="insight-card" style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-left: 4px solid {PRIMARY_COLOR};">
        <h3 style="color: {PRIMARY_COLOR}; font-size: 1.3rem; margin-bottom: 10px;">🌏 National Overview</h3>
        <p style="color: {TEXT_COLOR}; font-size: 1.05rem; line-height: 1.6; margin-bottom: 10px;">
            The national average CPI for {selected_commodity} in {selected_year} is 
            <span style="font-weight: bold; color: {PRIMARY_COLOR}; font-size: 1.2rem;">{national_avg:.2f}</span>
            with a standard deviation of {national_std:.2f}. 
            <span style="font-weight: bold; color: {ACCENT_COLOR}">{regions_above_avg} out of {total_regions} regions</span>
            have prices above the national average.
        </p>
        <p style="color: {TEXT_COLOR}; font-size: 1.05rem;">
            Year-over-year change: 
            <span style="font-weight: bold; color: {change_color}; font-size: 1.15rem;">{change_text}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Insight 2: Regional Disparities
    price_range = filtered['CPI'].max() - filtered['CPI'].min()
    price_disparity_pct = (price_range / national_avg) * 100
    
    # Find regions with extreme prices
    best_region = filtered.loc[filtered['CPI'].idxmin(), 'Geolocation']
    worst_region = filtered.loc[filtered['CPI'].idxmax(), 'Geolocation']
    best_cpi = filtered['CPI'].min()
    worst_cpi = filtered['CPI'].max()
    
    st.markdown(f"""
    <div class="insight-card" style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-left: 4px solid {ACCENT_COLOR};">
        <h3 style="color: {PRIMARY_COLOR}; font-size: 1.3rem; margin-bottom: 10px;">🏘️ Regional Price Disparities</h3>
        <p style="color: {TEXT_COLOR}; font-size: 1.05rem; line-height: 1.6; margin-bottom: 10px;">
            There is a 
            <span style="font-weight: bold; color: {BAD_COLOR if price_disparity_pct > 20 else ACCENT_COLOR}; font-size: 1.2rem;">{price_disparity_pct:.1f}%</span>
            price variation across regions. The most affordable prices are in 
            <span style="font-weight: bold; color: {GOOD_COLOR}">{best_region}</span>
            (CPI: {best_cpi:.2f}), while the highest prices are in 
            <span style="font-weight: bold; color: {BAD_COLOR}">{worst_region}</span>
            (CPI: {worst_cpi:.2f}).
        </p>
        <p style="color: {TEXT_COLOR}; font-size: 1rem;">
            💡 Recommendation: 
            <span style="font-style: italic;">
                {
                    "High price disparity suggests potential market inefficiencies or transportation costs." 
                    if price_disparity_pct > 20 
                    else "Price consistency across regions indicates a well-integrated market."
                }
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Insight 3: Historical Trend Analysis
    if len(historical['Year'].unique()) > 3:
        # Calculate 5-year trend if data available
        recent_years = sorted(historical['Year'].unique())[-5:]
        trend_data = historical[historical['Year'].isin(recent_years)].groupby('Year')['CPI'].mean()
        
        # Simple trend calculation
        if len(trend_data) > 1:
            years = np.array(trend_data.index)
            prices = np.array(trend_data.values)
            # Calculate linear regression
            n = len(years)
            slope = (n * sum(years * prices) - sum(years) * sum(prices)) / (n * sum(years**2) - sum(years)**2)
            annual_change_pct = (slope / prices[0]) * 100
            
            trend_direction = "increasing" if slope > 0 else "decreasing"
            trend_color = BAD_COLOR if slope > 0 else GOOD_COLOR
            
            st.markdown(f"""
            <div class="insight-card" style="background: linear-gradient(135deg, #ddd6fe 0%, #c7d2fe 100%); border-left: 4px solid #8b5cf6;">
                <h3 style="color: {PRIMARY_COLOR}; font-size: 1.3rem; margin-bottom: 10px;">📈 Historical Trend Analysis</h3>
                <p style="color: {TEXT_COLOR}; font-size: 1.05rem; line-height: 1.6; margin-bottom: 10px;">
                    Over the past {len(recent_years)} years, {selected_commodity} prices have been 
                    <span style="font-weight: bold; color: {trend_color}">{trend_direction}</span>
                    at an average annual rate of 
                    <span style="font-weight: bold; color: {trend_color}; font-size: 1.2rem;">{abs(annual_change_pct):.1f}%</span>.
                    At this rate, prices could {
                        "rise" if slope > 0 else "fall"
                    } by approximately {abs(slope * 3):.2f} points over the next 3 years.
                </p>
                <p style="color: {TEXT_COLOR}; font-size: 1rem;">
                    📊 Price Volatility: 
                    <span style="font-weight: bold; color: {
                        BAD_COLOR if national_std/national_avg > 0.15 
                        else ACCENT_COLOR if national_std/national_avg > 0.08 
                        else GOOD_COLOR
                    }">
                        {
                            "High" if national_std/national_avg > 0.15 
                            else "Moderate" if national_std/national_avg > 0.08 
                            else "Low"
                        }
                    </span>
                    (CV: {(national_std/national_avg)*100:.1f}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Insight 4: Region-specific insights (if a region is selected)
    if selected_region and selected_region in filtered['Geolocation'].values:
        region_data = filtered[filtered['Geolocation'] == selected_region]
        region_cpi = region_data['CPI'].iloc[0]
        region_rank = (filtered['CPI'] < region_cpi).sum() + 1
        
        # Compare with neighboring regions or similar regions
        all_commodities = df[(df["Year"] == selected_year) & (df["Geolocation"] == selected_region)]
        expensive_items = all_commodities.nlargest(3, 'CPI')['Commodity'].tolist()
        affordable_items = all_commodities.nsmallest(3, 'CPI')['Commodity'].tolist()
        
        st.markdown(f"""
        <div class="insight-card" style="background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%); border-left: 4px solid #ec4899;">
            <h3 style="color: {PRIMARY_COLOR}; font-size: 1.3rem; margin-bottom: 10px;">🏛️ {selected_region} Analysis</h3>
            <p style="color: {TEXT_COLOR}; font-size: 1.05rem; line-height: 1.6; margin-bottom: 10px;">
                {selected_region} ranks 
                <span style="font-weight: bold; color: {PRIMARY_COLOR}; font-size: 1.1rem;">#{region_rank} out of {total_regions}</span>
                for {selected_commodity} prices. The region's CPI is 
                <span style="font-weight: bold; color: {
                    BAD_COLOR if region_cpi > national_avg else GOOD_COLOR
                }">
                    {abs((region_cpi - national_avg)/national_avg * 100):.1f}% {
                        "above" if region_cpi > national_avg else "below"
                    } the national average
                </span>.
            </p>
            <div>
                <p style="color: {TEXT_COLOR}; font-size: 0.95rem; margin-bottom: 5px;">
                    🔴 Most expensive items: 
                    <span style="font-style: italic;">{", ".join(expensive_items[:3])}</span>
                </p>
                <p style="color: {TEXT_COLOR}; font-size: 0.95rem;">
                    🟢 Most affordable items: 
                    <span style="font-style: italic;">{", ".join(affordable_items[:3])}</span>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Main application layout
with tab1:  # Dashboard tab
    # Show KPI cards
    create_kpi_cards(selected_year, selected_commodity, st.session_state.selected_region)
    
    # Show rank visualization if a region is selected
    if st.session_state.selected_region:
        create_rank_visualization(selected_year, selected_commodity, st.session_state.selected_region)
        st.markdown("---")
    
    # Map and Radar chart row
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Create and display map
        map_fig, filtered_data = create_map(selected_year, selected_commodity, st.session_state.selected_region)
        
        # Create a selectbox for region selection as an alternative to map clicks
        regions = sorted(filtered_data['Geolocation'].unique())
        selected_region_index = 0
        if st.session_state.selected_region in regions:
            selected_region_index = regions.index(st.session_state.selected_region)
        
        selected_region_from_dropdown = st.selectbox(
            "Select a region:",
            options=regions,
            index=selected_region_index,
            key="region_selector"
        )
        
        # Update session state if region selection changes
        if selected_region_from_dropdown != st.session_state.selected_region:
            st.session_state.selected_region = selected_region_from_dropdown
            st.rerun()
        
        # Display the map
        st.plotly_chart(map_fig, use_container_width=True)
    
    with col2:
        # Commodity comparison using radar chart
        st.markdown(f"<h3 style='text-align: center; color: {PRIMARY_COLOR}; font-weight: bold; margin-bottom: 10px;'>Commodity Comparison</h3>", unsafe_allow_html=True)
        radar_fig = create_radar_chart(selected_year, selected_commodity, st.session_state.selected_region)
        st.plotly_chart(radar_fig, use_container_width=True)
    
    # Trend graph
    st.markdown("---")
    trend_fig = create_trend_graph(selected_commodity, st.session_state.selected_region)
    st.plotly_chart(trend_fig, use_container_width=True)

with tab2:  # Insights tab
    # Generate insights
    generate_insights(selected_year, selected_commodity, st.session_state.selected_region) 
