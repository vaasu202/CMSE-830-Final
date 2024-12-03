import pandas as pd
import numpy as np
import plotly.express as px
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.vizier import Vizier
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def about_section():
    st.sidebar.title("About This Project")
    st.sidebar.markdown(
        """
        # Exoplanet & Stellar Analysis App 🚀
        This project aims to analyze and predict the equilibrium temperatures of exoplanets based on their physical characteristics and stellar properties.

        ## Key Features:
        - **Data Collection**: We gather data from NASA's Exoplanet Archive, combining exoplanetary and stellar information.
        - **Exploratory Data Analysis (EDA)**: Visualize distributions, correlations, and other relationships in the data to understand key trends.
        - **Modeling**: Build predictive models using Random Forest and Gradient Boosting algorithms to estimate the equilibrium temperature of exoplanets.
        - **Prediction**: Based on user input, make predictions about the equilibrium temperature of exoplanets using the trained models.
        
        ## Models Used:
        - **Random Forest Regressor**: An ensemble method that builds multiple decision trees to make predictions based on the features provided.
        - **Gradient Boosting Regressor**: Another powerful ensemble method that builds decision trees sequentially, optimizing performance with each tree.

        This application provides an intuitive platform for astronomers, researchers, and enthusiasts to explore exoplanet data and make temperature predictions based on stellar and planetary characteristics. 🌌

        *Made with ❤️ by Vaasu Sohee*
        """
    )

# ---------------------------
# 1. Data Collection and Preparation
# ---------------------------
@st.cache_resource(ttl=86400)
def download_exoplanet_data():
    exoplanet_columns = ['pl_name', 'hostname', 'pl_orbper', 'pl_rade', 'pl_masse', 'pl_eqt']
    exoplanets = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select=exoplanet_columns
    )
    return exoplanets.to_pandas()

@st.cache_resource(ttl=86400)
def download_stellar_data():
    stellar_columns = ['hostname', 'st_spectype', 'st_lum', 'st_teff', 'st_met', 'st_age']
    stellar = NasaExoplanetArchive.query_criteria(
        table="stellarhosts",
        select=stellar_columns
    )
    return stellar.to_pandas()

@st.cache_resource(ttl=86400)
def download_2mass_data():
    Vizier.ROW_LIMIT = -1
    result = Vizier.query_catalog("II/246", catalog=["2MASS"])
    data = result[0].to_pandas()
    # Filter or process 2MASS data as needed
    return data

@st.cache_resource(ttl=86400)
def clean_and_merge_data(exoplanets, stellar):
    exoplanets_clean = exoplanets.dropna(subset=['pl_name', 'hostname', 'pl_orbper', 'pl_rade', 'pl_masse'])
    stellar_clean = stellar.dropna(subset=['hostname', 'st_spectype', 'st_lum', 'st_teff'])
    combined_df = pd.merge(exoplanets_clean, stellar_clean, on='hostname', how='inner')
    combined_df['pl_eqt'] = combined_df['pl_eqt'].fillna(combined_df['pl_eqt'].mean())
    return combined_df

# ---------------------------
# Add 2MASS Data Integration
# ---------------------------
@st.cache_resource(ttl=86400)
def integrate_2mass_data(combined_df, mass_data):
    """
    Merge 2MASS catalog data with the combined exoplanet and stellar data.
    """
    # Assume `2MASS` data contains `hostname` or similar matching column
    integrated_df = pd.merge(combined_df, mass_data, left_on='hostname', right_on='2MASS_hostname', how='left')
    return integrated_df

# ---------------------------
# 2. Exploratory Data Analysis (EDA)
# ---------------------------
def plot_visualizations(data):
    st.subheader("Choose a Visualization")
    viz_options = [
        "Exoplanet Radii Distribution",
        "Mass vs Radius Scatter Plot",
        "Stellar Luminosity by Spectral Type",
        "Correlation Heatmap",
        "Scatter Matrix (Top Features)"
    ]
    choice = st.selectbox("Visualization Options", viz_options)

    if choice == "Exoplanet Radii Distribution":
        fig = px.histogram(
            data,
            x='pl_rade',
            nbins=30,
            title="Exoplanet Radii Distribution",
            color_discrete_sequence=["#636EFA"],
            height=600,
            width=900
        )
        st.plotly_chart(fig)

    elif choice == "Mass vs Radius Scatter Plot":
        fig = px.scatter(
            data,
            x='pl_rade',
            y='pl_masse',
            color='st_spectype',
            title="Exoplanet Mass vs Radius",
            height=600,
            width=900
        )
        st.plotly_chart(fig)

    elif choice == "Stellar Luminosity by Spectral Type":
        fig = px.box(
            data,
            x='st_spectype',
            y='st_lum',
            title="Stellar Luminosity Distribution",
            color_discrete_sequence=["#EF553B"],
            height=600,
            width=900
        )
        st.plotly_chart(fig)

    elif choice == "Correlation Heatmap":
        numeric_data = data.select_dtypes(include=[np.number])
        corr = numeric_data.corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="Viridis",
            title="Correlation Heatmap",
            height=600,
            width=900
        )
        st.plotly_chart(fig)

    elif choice == "Scatter Matrix (Top Features)":
        fig = px.scatter_matrix(
            data,
            dimensions=['pl_rade', 'pl_masse', 'st_teff', 'st_lum'],
            color='st_spectype',
            title="Scatter Matrix",
            height=800,
            width=900
        )
        st.plotly_chart(fig)

# ---------------------------
# 3. Model Development and Evaluation
# ---------------------------
@st.cache_resource(ttl=86400)
def train_models(data):
    X = data[['pl_rade', 'pl_masse', 'st_teff', 'st_lum']]
    y = data['pl_eqt']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)

    def calculate_accuracy(y_true, y_pred):
        percentage_errors = 1 - (abs(y_true - y_pred) / y_true)
        return np.mean(percentage_errors[~np.isnan(percentage_errors)]) * 100

    rf_metrics = {
        "MSE": mean_squared_error(y_test, rf_pred),
        "R²": r2_score(y_test, rf_pred),
        "MAE": mean_absolute_error(y_test, rf_pred),
        "Accuracy (%)": calculate_accuracy(y_test, rf_pred),
    }

    gb_metrics = {
        "MSE": mean_squared_error(y_test, gb_pred),
        "R²": r2_score(y_test, gb_pred),
        "MAE": mean_absolute_error(y_test, gb_pred),
        "Accuracy (%)": calculate_accuracy(y_test, gb_pred),
    }

    return rf_model, rf_metrics, gb_model, gb_metrics

# ---------------------------
# 4. Prediction Section
# ---------------------------
def predict_with_model(model, ranges):
    st.subheader("🌌 Predict Exoplanet Equilibrium Temperature")
    
    # Slider inputs
    pl_rade = st.slider("Exoplanet Radius (in Earth radii)", float(ranges['pl_rade'][0]), float(ranges['pl_rade'][1]), step=0.1)
    pl_masse = st.slider("Exoplanet Mass (in Earth masses)", float(ranges['pl_masse'][0]), float(ranges['pl_masse'][1]), step=0.1)
    st_teff = st.slider("Stellar Temperature (in Kelvin)", float(ranges['st_teff'][0]), float(ranges['st_teff'][1]), step=10.0)
    st_lum = st.slider("Stellar Luminosity (in Solar units)", float(ranges['st_lum'][0]), float(ranges['st_lum'][1]), step=0.1)

    # Add custom CSS to style the submit button
    st.markdown("""
        <style>
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 10px 24px;
                border: none;
                border-radius: 50px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #45a049;
                cursor: pointer;
                transform: translateY(-3px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
            }
        </style>
    """, unsafe_allow_html=True)

    # Initialize prediction variable
    prediction = None

    # Add a stylish Submit button
    if st.button("Submit Prediction"):
        # Feature array for prediction
        features = np.array([[pl_rade, pl_masse, st_teff, st_lum]])
        prediction = model.predict(features)[0]

        # Trigger balloons for celebration
        st.balloons()

    # Display predicted result only if prediction exists
    if prediction is not None:
        st.markdown(
            f"""
            <div style='
                background: linear-gradient(135deg, #f0f4f8, #d6e1f5);
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
                text-align: center;
            '>
                <h2 style='color: #4CAF50; font-size: 28px;'>🌠 Prediction Result</h2>
                <h3 style='color: #2196F3; font-size: 24px;'>Predicted Exoplanet Equilibrium Temperature: <span style="font-weight: bold;">{prediction:.2f} K</span></h3>
                <p style='font-size: 16px; color: #555;'>This prediction is based on the provided exoplanet radius, mass, and stellar parameters. 🌌</p>
            </div>
            """, unsafe_allow_html=True
        )

# ---------------------------
# Main Streamlit App
# ---------------------------
def main():
    st.set_page_config(page_title="Exoplanet Analysis", layout="wide")
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>Exoplanet & Stellar Analysis App 🚀</h1>",
        unsafe_allow_html=True,
    )

    # Data download and merging
    exoplanets = download_exoplanet_data()
    stellar = download_stellar_data()
    combined_df = clean_and_merge_data(exoplanets, stellar)

    st.sidebar.title("Navigation")
    options = ["View Data", "EDA", "Modeling", "Prediction"]
    choice = st.sidebar.radio("Go to", options)

    about_section()

    # View data section
    if choice == "View Data":
        st.header("View Dataset")
        st.write(combined_df)

    # EDA section
    elif choice == "EDA":
        st.header("Exploratory Data Analysis")
        plot_visualizations(combined_df)

    # Modeling section
    elif choice == "Modeling":
        st.header("Model Development and Evaluation")

        st.markdown(
            """
            **Metrics Explained:**
            - **MSE (Mean Squared Error):** Measures the average squared difference between actual and predicted values. Lower is better.
            - **R² (Coefficient of Determination):** Measures how well the model explains the variance in the target variable. Higher is better.
            - **MAE (Mean Absolute Error):** Average of absolute differences between actual and predicted values. Lower is better.
            - **Accuracy (%):** Average percentage accuracy of predictions relative to actual values.
            """
        )
        rf_model, rf_metrics, gb_model, gb_metrics = train_models(combined_df)
        
        st.subheader("📊 Random Forest Metrics")
        st.write(rf_metrics)


        st.subheader("📉 Gradient Boosting Metrics")
        st.write(gb_metrics)

        st.subheader(" 🙇🏻‍♂️ Comparing the two")
        st.markdown(
            """
            **By comparison, the Random Forerst worked way better than Gradient Boosting ensemble**
            """
        )

        # Visualize Model Predictions
        st.subheader("Prediction vs. Actual Plot")
        X = combined_df[['pl_rade', 'pl_masse', 'st_teff', 'st_lum']]
        y = combined_df['pl_eqt']
        rf_pred = rf_model.predict(X)
        fig = px.scatter(
            x=y,
            y=rf_pred,
            labels={'x': 'Actual', 'y': 'Predicted'},
            title="Random Forest: Actual vs Predicted Equilibrium Temperature",
            height=600,
            width=900
        )
        st.plotly_chart(fig)

    # Prediction section
    elif choice == "Prediction":
        rf_model, _, _, _ = train_models(combined_df)
        ranges = {
            'pl_rade': (combined_df['pl_rade'].min(), combined_df['pl_rade'].max()),
            'pl_masse': (combined_df['pl_masse'].min(), combined_df['pl_masse'].max()),
            'st_teff': (combined_df['st_teff'].min(), combined_df['st_teff'].max()),
            'st_lum': (combined_df['st_lum'].min(), combined_df['st_lum'].max())
        }
        predict_with_model(rf_model, ranges)

if __name__ == "__main__":
    main()
    
