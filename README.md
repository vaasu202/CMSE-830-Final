# Exoplanet and Stellar Properties Analysis

This Streamlit-based application analyzes the relationship between exoplanet characteristics and their host stars. The data is sourced from the NASA Exoplanet Archive via the `astroquery` library. The application allows for data visualization and exploration of the underlying relationships between stellar and exoplanet parameters.

## Features

- **Data Download & Caching**: Automatically fetches exoplanet and stellar data from the NASA Exoplanet Archive and caches it for 24 hours to reduce API calls.
- **Data Cleaning & Merging**: Cleans the data by removing rows with missing values and merges exoplanet and stellar data on the common field `hostname`.
- **Exploratory Data Analysis (EDA)**: Provides various interactive visualizations to explore key exoplanet and stellar features.
- **Insights & Observations**: Displays statistical insights about the relationships between exoplanet and stellar properties.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/exoplanet-analysis.git
    cd exoplanet-analysis
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Data Sources

This project utilizes data from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/). The following datasets are used:

- **Exoplanet Data**: Includes exoplanet name, orbital period, radius, mass, and equilibrium temperature.
- **Stellar Data**: Includes host star spectral type, luminosity, temperature, and age.

## Application Structure

1. **View Data**: Displays the raw exoplanet and stellar datasets, as well as the cleaned and merged dataset.
2. **Data Cleaning & Merging**: Removes records with missing essential values and merges the exoplanet and stellar datasets based on the `hostname` field.
3. **Exploratory Data Analysis (EDA)**: Offers a variety of visualizations such as histograms, scatter plots, and heatmaps to explore relationships between variables.
4. **Insights & Observations**: Presents key insights derived from the analysis, including correlation coefficients and average exoplanet characteristics by stellar type.

## Visualizations

- **Exoplanet Radius Distribution**: Histogram of exoplanet radii.
- **Exoplanet Mass vs. Radius**: Scatter plot of exoplanet mass versus radius, color-coded by spectral type.
- **Stellar Luminosity by Spectral Type**: Box plot showing stellar luminosity distributions by spectral type.
- **Correlation Heatmap**: Displays the correlation between selected features.
- **Stellar Age vs. Luminosity**: Scatter plot of stellar age versus luminosity, color-coded by spectral type.

## Dependencies

- **Pandas**: For data manipulation and cleaning.
- **Numpy**: For numerical operations.
- **Plotly**: For interactive visualizations.
- **Streamlit**: For building the web-based app.
- **Astroquery**: For querying the NASA Exoplanet Archive.

Install the dependencies with:

```bash
pip install pandas numpy plotly streamlit astroquery
