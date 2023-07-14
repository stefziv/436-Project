import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression

if st.session_state.lm:
    model = st.session_state.lm
    coefficients = model.coef_ 
    df = st.session_state.df
    X, y = st.session_state.lm_params
    y_pred = model.predict(X)
    sensitivity_results = []

    for coeff_index in range(coefficients.shape[0]):
        vary_coeff_values = np.linspace(np.min(coefficients[coeff_index]), np.max(coefficients[coeff_index]), 100)

        coefficients_sensitivity = coefficients.copy()

        for _, coeff_value in enumerate(vary_coeff_values):
            coefficients_sensitivity[coeff_index] = coeff_value
            model_sensitivity = LinearRegression()
            model_sensitivity.fit(X,y)
            model_sensitivity.coef_ = coefficients_sensitivity
            y_pred_sensitivity = model_sensitivity.predict(X)
            sensitivity_diff = y_pred_sensitivity - y_pred
            sensitivity_results.append(sensitivity_diff)

    sensitivity_results = np.array(sensitivity_results)

    fig = go.Figure()

    for i in range(sensitivity_results.shape[1]):
        fig.add_trace(go.Scatter(
            x=np.tile(vary_coeff_values, coefficients.shape[0]),
            y=sensitivity_results[:, i],
            mode='lines',
            name=f'Coefficient {i + 1} sensitivity'
        ))

    fig.update_layout(
        xaxis_title='Varying Coefficient Value',
        yaxis_title='Sensitivity',
        title='Sensitivity Analysis of Coefficients'
    )

    fig.show()
