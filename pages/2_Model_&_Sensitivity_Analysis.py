\import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

if st.session_state.lm:
    model = st.session_state.lm
    df = st.session_state.df
    X,y = st.session_state.lm_params
    coeffs = model.coef_ 
    coeff_df = pd.DataFrame(
        data=coeffs,
        columns=["coefficients"],
        index=df.columns
    )
    baseline_predictions = model.predict(X)
    sensitivity_df = pd.DataFrame(columns=['Feature', 'Baseline', 'Perturbed', 'Difference'])
    sensitivity_l = []
    for feature in X.columns:
        perturbed_X = X.copy()
        perturbed_X[feature] = np.random.normal(perturbed_X[feature].mean()*3,
                                                perturbed_X[feature].std()*3, 
                                                len(perturbed_X))

        perturbed_predictions = model.predict(perturbed_X)
        difference = perturbed_predictions - baseline_predictions
        sensitivity_l.append({
            'Feature': feature,
            'Baseline': np.mean(baseline_predictions),
            'Perturbed': np.mean(perturbed_predictions),
            'Difference': np.mean(difference)
        })
    
    sensitivity_df = pd.concat([sensitivity_df, pd.DataFrame(sensitivity_l)], ignore_index=True)
    st.dataframe(sensitivity_df)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Positive',
        x=sensitivity_df['Perturbed'],
        y=sensitivity_df['Feature'],
        marker_color='green',
        orientation='h'
    ))

    fig.update_layout(
        title='Sensitivity Analysis',
        xaxis_title='Predicted Output',
        yaxis_title='Feature',
        barmode='relative',
        bargap=0.2,
        bargroupgap=0.1,
        showlegend=False
    )
    st.plotly_chart(fig)
    df_corr = X.copy()
    df_corr["SalePrice"] = y
    _ = df_corr.columns.tolist()
    cols = _[-1:] + _[:-1]
    df_corr=df_corr[cols]
    st.plotly_chart(px.imshow(df_corr.corr(),
                            width = 700,
                            height = 700,
                            color_continuous_scale='Viridis'
                            )
                    )
    
