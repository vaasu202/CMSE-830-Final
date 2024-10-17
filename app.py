import pandas as pd
import numpy as np
import plotly.express as px
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import streamlit as st

@st.cache_resource(ttl=86400)  # Cache data for 1 day (86400 seconds)
def download_exoplanet_data_astroquery():
    try:
        exoplanet_columns = ['pl_name', 'hostname', 'pl_orbper', 'pl_rade', 'pl_masse', 'pl_eqt']
        exoplanets = NasaExoplanetArchive.query_criteria(
            table="pscomppars",
            select=exoplanet_columns
        )
        exoplanets_df = exoplanets.to_pandas()
        return exoplanets_df
    except Exception as e:
        st.error(f"Error downloading exoplanet data via Astroquery: {e}")
        st.stop()

@st.cache_resource(ttl=86400)
def download_stellar_properties_astroquery():
    try:
        stellar_columns = ['hostname', 'st_spectype', 'st_lum', 'st_teff', 'st_met', 'st_age']
        stellar = NasaExoplanetArchive.query_criteria(
            table="stellarhosts",  
            select=stellar_columns
        )
        stellar_df = stellar.to_pandas()
        return stellar_df
    except Exception as e:
        st.error(f"Error downloading stellar properties data via Astroquery: {e}")
        st.stop()

# ---------------------------
# 2. Data Cleaning and Merging
# ---------------------------

@st.cache_resource(ttl=86400)
def clean_and_merge_data(exoplanets, stellar):
    essential_exo_cols = ['pl_name', 'hostname', 'pl_orbper', 'pl_rade', 'pl_masse']
    exoplanets_clean = exoplanets.dropna(subset=essential_exo_cols)

    essential_stellar_cols = ['hostname', 'st_spectype', 'st_lum', 'st_teff']
    stellar_clean = stellar.dropna(subset=essential_stellar_cols)

    combined_df = pd.merge(exoplanets_clean, stellar_clean, on='hostname', how='inner')

    return combined_df

# ---------------------------
# 3. Exploratory Data Analysis (EDA)
# ---------------------------

def plot_histogram(data, column, title, xlabel):
    fig = px.histogram(data, x=column, title=title, labels={column: xlabel}, nbins=30, marginal='box')
    fig.update_layout(height=600, width=900)  
    st.plotly_chart(fig, use_container_width=True)

def plot_scatter(data, x, y, color, title, xlabel, ylabel):
    fig = px.scatter(data, x=x, y=y, color=color, title=title,
                     labels={x: xlabel, y: ylabel},
                     hover_name='pl_name',  
                     opacity=0.7)
    fig.update_layout(height=600, width=900)  
    st.plotly_chart(fig, use_container_width=True)

def plot_boxplot(data, x, y, title, xlabel, ylabel):
    fig = px.box(data, x=x, y=y, title=title,
                 labels={x: xlabel, y: ylabel})
    fig.update_layout(height=600, width=900)  
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_heatmap(data, columns, title):
    corr = data[columns].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis', title=title)
    fig.update_layout(height=600, width=900)  
    st.plotly_chart(fig, use_container_width=True)

def plot_pairplot(data, features, color):
    fig = px.scatter_matrix(data, dimensions=features, color=color, 
                            title='Pair Plot of Selected Features',
                            height=900, width=900)  
    st.plotly_chart(fig, use_container_width=True)

def plot_countplot(data, x, title, xlabel, ylabel):
    fig = px.histogram(data, x=x, title=title, labels={x: xlabel}, barmode='overlay')
    fig.update_layout(height=600, width=900)  
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# 4. Streamlit App Layout
# ---------------------------

def main():
    st.set_page_config(page_title="Exoplanet and Stellar Properties Analysis", layout='wide')  # Set page config
    st.title("Exoplanet and Stellar Properties Analysis")
    st.markdown("""This application analyzes the relationship between exoplanet characteristics and their host stars. Data is sourced from the NASA Exoplanet Archive and includes various parameters for both exoplanets and stars.""")

    st.sidebar.title("Navigation")
    options = ["View Data", "Data Cleaning & Merging", "Exploratory Data Analysis", "Insights", "Creator"]
    choice = st.sidebar.radio("Go to", options)

    if choice == "Creator":
        st.header("😁 The person behind this all")
        st.subheader("Vaasu Sohee")
        st.write("Graduate Student at Michigan State University 🟢")

    
    exoplanets = download_exoplanet_data_astroquery()
    stellar = download_stellar_properties_astroquery()
    combined_df = clean_and_merge_data(exoplanets, stellar)

    if choice == "View Data":
        st.header("📊 View Datasets")
        st.subheader("Exoplanet Dataset")
        st.dataframe(exoplanets)
        
        st.subheader("Stellar Properties Dataset")
        st.dataframe(stellar)
        
        st.subheader("Combined Dataset")
        st.dataframe(combined_df)

    elif choice == "Data Cleaning & Merging":
        st.header("🧹 Data Cleaning & Merging")
        st.write("""The datasets have been cleaned by removing records with missing essential values. They are then merged on the common key `hostname` to create a combined dataset for analysis.""")
        st.subheader("Missing Values After Cleaning")
        st.write("**Exoplanet Dataset:**")
        st.write(exoplanets.drop(columns=['pl_name', 'hostname', 'pl_orbper', 'pl_rade', 'pl_masse'], errors='ignore').isnull().sum())
        
        st.write("**Stellar Properties Dataset:**")
        st.write(stellar.drop(columns=['hostname', 'st_spectype', 'st_lum', 'st_teff'], errors='ignore').isnull().sum())
        
        st.subheader("Combined Dataset Preview")
        st.dataframe(combined_df.head())

    elif choice == "Exploratory Data Analysis":
        st.header("🔍 Exploratory Data Analysis (EDA)")
        
        
        eda_options = [
            "Distribution of Exoplanet Radii",
            "Exoplanet Mass vs. Radius by Spectral Type",
            "Stellar Luminosity by Spectral Type",
            "Correlation Heatmap",
            "Distribution of Exoplanet Orbital Periods",
            "Equilibrium Temperature vs. Stellar Effective Temperature",
            "Pair Plot of Selected Features",
            "Age Distribution of Host Stars",
            "Exoplanet Radius vs. Stellar Luminosity",
            "Missing Values Heatmap"
        ]
        selected_eda = st.selectbox("Select EDA Plot", eda_options)
        
        if selected_eda == "Distribution of Exoplanet Radii":
            st.subheader("Distribution of Exoplanet Radii")
            plot_histogram(combined_df, 'pl_rade', 'Distribution of Exoplanet Radii', 'Radius (Earth Radii)')
        
        elif selected_eda == "Exoplanet Mass vs. Radius by Spectral Type":
            st.subheader("Exoplanet Mass vs. Radius by Spectral Type")
            plot_scatter(combined_df, 'pl_rade', 'pl_masse', 'st_spectype', 
                         'Exoplanet Mass vs. Radius by Spectral Type', 
                         'Radius (Earth Radii)', 'Mass (Earth Masses)')
        
        elif selected_eda == "Stellar Luminosity by Spectral Type":
            st.subheader("Stellar Luminosity by Spectral Type")
            plot_boxplot(combined_df, 'st_spectype', 'st_lum', 
                        'Stellar Luminosity by Spectral Type', 
                        'Spectral Type', 'Luminosity (Solar Luminosities)')
        
        elif selected_eda == "Correlation Heatmap":
            st.subheader("Correlation Heatmap of Selected Features")
            selected_columns = ['pl_orbper', 'pl_rade', 'pl_masse', 'pl_eqt', 
                                'st_lum', 'st_teff', 'st_met', 'st_age']
            plot_correlation_heatmap(combined_df, selected_columns, 'Correlation Heatmap of Selected Features')
        
        elif selected_eda == "Distribution of Exoplanet Orbital Periods":
            st.subheader("Distribution of Exoplanet Orbital Periods")
            plot_histogram(combined_df, 'pl_orbper', 'Distribution of Exoplanet Orbital Periods', 
                          'Orbital Period (days)')
        
        elif selected_eda == "Equilibrium Temperature vs. Stellar Effective Temperature":
            st.subheader("Equilibrium Temperature vs. Stellar Effective Temperature")
            plot_scatter(combined_df, 'st_teff', 'pl_eqt', 'st_spectype', 
                         'Exoplanet Equilibrium Temperature vs. Stellar Effective Temperature', 
                         'Stellar Effective Temperature (K)', 'Exoplanet Equilibrium Temperature (K)')
        
        elif selected_eda == "Pair Plot of Selected Features":
            st.subheader("Pair Plot of Selected Features")
            selected_features = ['pl_rade', 'pl_masse', 'pl_orbper', 'st_lum', 'st_teff']
            plot_pairplot(combined_df, selected_features, 'st_spectype')
        
        elif selected_eda == "Age Distribution of Host Stars":
            st.subheader("Age Distribution of Host Stars")
            plot_histogram(combined_df, 'st_age', 'Age Distribution of Host Stars', 
                          'Age (Gyr)')
        
        elif selected_eda == "Exoplanet Radius vs. Stellar Luminosity":
            st.subheader("Exoplanet Radius vs. Stellar Luminosity")
            plot_scatter(combined_df, 'st_lum', 'pl_rade', 'st_spectype', 
                         'Exoplanet Radius vs. Stellar Luminosity', 
                         'Stellar Luminosity (Solar Luminosities)', 'Exoplanet Radius (Earth Radii)')
        
        elif selected_eda == "Missing Values Heatmap":
            st.subheader("Missing Values Heatmap After Cleaning")
            fig = px.imshow(combined_df.isnull(), color_continuous_scale='viridis', title='Missing Values Heatmap After Cleaning')
            fig.update_layout(height=600, width=900)  
            st.plotly_chart(fig, use_container_width=True)

    elif choice == "Insights":
        st.header("💡 Insights and Observations")
        
        st.subheader("Correlation between Exoplanet Mass and Radius")
        mass_radius_corr = combined_df['pl_masse'].corr(combined_df['pl_rade'])
        st.write(f"**Correlation coefficient:** {mass_radius_corr:.2f}")
        
        st.subheader("Average Exoplanet Radius by Spectral Type")
        avg_radius_by_spectral = combined_df.groupby('st_spectype')['pl_rade'].mean().sort_values()
        st.table(avg_radius_by_spectral.rename("Average Radius (Earth Radii)"))
        
        st.subheader("Number of Exoplanets by Host Star Spectral Type")
        plot_countplot(combined_df, 'st_spectype', 
                      'Number of Exoplanets by Host Star Spectral Type', 
                      'Spectral Type', 'Number of Exoplanets')
        
        st.subheader("Exoplanet Radius Distribution by Host Star Spectral Type")
        fig = px.box(combined_df, x='st_spectype', y='pl_rade', 
                      title='Exoplanet Radius Distribution by Host Star Spectral Type')
        fig.update_layout(height=600, width=900) 
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Stellar Age vs. Luminosity by Spectral Type")
        plot_scatter(combined_df, 'st_age', 'st_lum', 'st_spectype', 
                     'Stellar Age vs. Luminosity by Spectral Type', 
                     'Age (Gyr)', 'Luminosity (Solar Luminosities)')

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""**Exoplanet and Stellar Properties Analysis**""")
    st.sidebar.markdown("""**Data Source:** NASA Exoplanet Archive**""")
    st.sidebar.markdown("""**Purpose:** Explore relationships between exoplanet characteristics and their host stars.""")
    st.sidebar.markdown("""**Developed with:** Streamlit, Pandas, Plotly, Astroquery""")
def get_table_columns_astroquery(table_name):
    try:
        query = f'SELECT * FROM {table_name} LIMIT 1'
        response = NasaExoplanetArchive.query_criteria(
            table=table_name,
            select='*'
        )
        df = response.to_pandas()
        return df.columns.tolist()
    except Exception as e:
        st.error(f"Error fetching columns for table `{table_name}`: {e}")
        st.stop()

if __name__ == "__main__":
    main()
