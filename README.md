# Exoplanet & Stellar Analysis App 🚀

This project aims to analyze and predict the equilibrium temperatures of exoplanets based on their physical characteristics and stellar properties.

## Table of Contents

1. [About This Project](#about-this-project)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Models & Metrics](#models--metrics)
5. [Contributors](#contributors)

## About This Project

This project focuses on building an application to analyze and predict the equilibrium temperatures of exoplanets using data from NASA's Exoplanet Archive. It combines exploratory data analysis (EDA), machine learning model training, and user-driven predictions.

### Key Features:
- **Data Collection**: We gather data from NASA's Exoplanet Archive, combining exoplanetary and stellar information.
- **Exploratory Data Analysis (EDA)**: Visualize distributions, correlations, and relationships in the data to understand key trends.
- **Modeling**: Build predictive models using Random Forest and Gradient Boosting algorithms to estimate the equilibrium temperature of exoplanets.
- **Prediction**: Based on user input, make predictions about the equilibrium temperature of exoplanets using the trained models.

### Models Used:
- **Random Forest Regressor**: An ensemble method that builds multiple decision trees to make predictions based on the features provided.
- **Gradient Boosting Regressor**: Another powerful ensemble method that builds decision trees sequentially, optimizing performance with each tree.

This application provides an intuitive platform for astronomers, researchers, and enthusiasts to explore exoplanet data and make temperature predictions based on stellar and planetary characteristics. 🌌

*Made with ❤️ by Vaasu Sohee*

---

## Installation

To set up this project on your local machine, follow these steps:

### Prerequisites:
- Python 3.8 or higher
- pip (Python package manager)

### Steps:
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/exoplanet-analysis.git
    cd exoplanet-analysis
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the exoplanet data from NASA's Exoplanet Archive by running the script.

4. Run the app using Streamlit:
    ```bash
    streamlit run app.py
    ```

---

## Usage

Once the app is running, you will have the following sections in the sidebar:

1. **View Data**: Displays the dataset containing exoplanet and stellar information.
2. **EDA**: Explore visualizations such as the distribution of exoplanet radii, mass vs radius scatter plot, and more.
3. **Modeling**: View metrics of the trained models, such as Mean Squared Error (MSE), R-squared (R²), Mean Absolute Error (MAE), and accuracy.
4. **Prediction**: Make predictions for the equilibrium temperature of exoplanets based on user inputs such as radius, mass, stellar temperature, and luminosity.

---

## Models & Metrics

### Models:
- **Random Forest Regressor**: Uses multiple decision trees to generate predictions. This model generally performs well in most cases.
- **Gradient Boosting Regressor**: Uses an ensemble of trees to reduce bias and variance, optimizing performance with each iteration.

### Metrics:
- **MSE (Mean Squared Error)**: Measures the average squared difference between actual and predicted values. Lower values are better.
- **R² (Coefficient of Determination)**: Measures how well the model explains the variance in the target variable. Higher values are better.
- **MAE (Mean Absolute Error)**: The average of absolute differences between actual and predicted values. Lower values are better.
- **Accuracy (%)**: Average percentage accuracy of predictions relative to actual values. Higher values indicate better model performance.

---

## Contributors

- **Vaasu Sohee** – Initial development and project setup.
  
If you'd like to contribute, feel free to fork the repository, make your changes, and submit a pull request!

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
