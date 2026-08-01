import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


MODEL_DIR = Path(__file__).parent / "Models"

st.set_page_config(
    page_title="Marketing Campaign Prediction",
    layout="wide"
)

model_path = Path(r"E:\SQL\Marketing Prediction\Models\best_regression_model.pkl")

print("File exists:", model_path.exists())
print("File size:", model_path.stat().st_size)
print("Modified:", model_path.stat().st_mtime)

#LOAD MODELS


regression_model = joblib.load(
    r"E:\SQL\Marketing Prediction\Models\best_regression_model.pkl"
)

classification_model = joblib.load(
    r"E:\SQL\Marketing Prediction\Models\best_classification_model.pkl"
)

classification_scaler = joblib.load(
    r"E:\SQL\Marketing Prediction\Models\classification_scaler.pkl"
)

label_encoder = joblib.load(
    r"E:\SQL\Marketing Prediction\Models\label_encoder.pkl"
)



#REGRESSION FEATURES

regression_features = [
    'duration',
    'impressions',
    'clicks',
    'leads',
    'conversions',
    'acquisition_cost',
    'engagement_score',
    'click_through_rate',
    'conversion_rate',
    'lead_conversion_rate',
    'cost_per_click',
    'campaign_type_Email',
    'campaign_type_Influencer',
    'campaign_type_Paid Ads',
    'campaign_type_SEO',
    'campaign_type_Social Media',
    'target_audience_College Students',
    'target_audience_Premium Shoppers',
    'target_audience_Tier 2 City Customers',
    'target_audience_Working Women',
    'target_audience_Youth',
    'brand_nykaa',
    'brand_purplle',
    'brand_tira',
    'language_Bengali',
    'language_English',
    'language_Hindi',
    'language_Tamil'
]



#SIDEBAR

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Revenue Prediction",
        "Profit Prediction"
    ]
)



#HOME

if page == "Home":


    st.image("E:\SQL\Marketing Prediction\Prediction.gif", width=800)

    st.title("Multi-Brand Marketing Analytics & Prediction System")

    st.write(
        """
        \nAn end-to-end Machine Learning application for analyzing,
        predicting, and optimizing marketing campaign performance
        across multiple beauty and e-commerce brands\n.
        """
    )
    st.divider()

    st.subheader("Project Description")

    st.write(
        """
        **Multi-Brand Marketing Analytics & Data Preprocessing Pipeline**

        Developed an end-to-end data cleaning, preprocessing, and
        exploratory data analysis (EDA) pipeline to transform raw,
        fragmented campaign data from brands such as Nykaa, Purplle,
        and Tira into structured, analytics-ready datasets.
        The project focuses on improving data reliability and generating
        actionable insights for performance marketing optimization.
        """
    )

    st.subheader("Core Challenges Addressed")

    st.markdown(
        """
        **🔹 Data Complexity**

        - Standardized raw marketing metrics including impressions,
          clicks, conversion rates, CAC, revenue, and ROI from
          multiple CSV sources.

        **🔹 Data Quality Issues**

        - Handled missing values, system-generated anomalies,
          outliers, and inconsistencies in ROI calculations.

        **🔹 Feature Complexity**

        - Transformed multi-valued categorical fields such as
          marketing channels into machine-learning-ready features.
        """
    )

    st.subheader("Key Implementations")

    st.markdown(
        """
        **🔹 Advanced Data Cleaning**

        - Engineered anomaly detection and outlier filtering techniques
          to remove incorrect system-generated values.

        **🔹 Feature Engineering**

        - Created derived business features including Profit/Loss
          indicators and marketing performance metrics.

        **🔹 Categorical Transformation**

        - Applied encoding techniques to handle multiple campaign
          categories and marketing channels.

        **🔹 Exploratory Data Analysis**

        - Built visual analysis to understand relationships between
          impressions, clicks, leads, conversions, revenue, and ROI.

        **🔹 Predictive Modeling**

        - Developed Machine Learning models for revenue prediction
          and campaign profitability classification.
        """
    )

    st.divider()

    st.divider()





#REVENUE PREDICTION


elif page == "Revenue Prediction":

    st.title("Revenue Prediction")

    st.write(
        "Enter campaign details below."
    )


    col1, col2 = st.columns(2)


    with col1:

        duration = st.number_input(
            "Duration in Days",
            min_value=1,
            value=30
        )


        impressions = st.number_input(
            "Impressions",
            min_value=0,
            value=50000
        )


        clicks = st.number_input(
            "Clicks",
            min_value=0,
            value=3000
        )


        leads = st.number_input(
            "Leads",
            min_value=0,
            value=1500
        )


        conversions = st.number_input(
            "Conversions",
            min_value=0,
            value=800
        )


        acquisition_cost = st.number_input(
            "Acquisition Cost",
            min_value=0.0,
            value=150.0
        )

        



    with col2:


        campaign_type = st.selectbox(
            "Campaign Type",
            [
                "Email",
                "Influencer",
                "Paid Ads",
                "SEO",
                "Social Media"
            ]
        )


        target_audience = st.selectbox(
            "Target Audience",
            [
                "College Students",
                "Premium Shoppers",
                "Tier 2 City Customers",
                "Working Women",
                "Youth"
            ]
        )


        brand = st.selectbox(
            "Brand",
            [
                "nykaa",
                "purplle",
                "tira"
            ]
        )


        language = st.selectbox(
            "Language",
            [
                "English",
                "Hindi",
                "Tamil",
                "Bengali"
            ]
        )

    predict_button = st.button(
        "Predict Revenue"
    )

    if predict_button:

     
#FEATURE ENGINEERING
      
        engagement_score = (((clicks + leads + conversions) / impressions) * 100
                    if impressions > 0 else 0
                )
        engagement_score = round(engagement_score, 2)

        total_acquisition_cost = acquisition_cost * conversions

        click_through_rate = (
            clicks / impressions
            if impressions > 0 else 0
        )

        conversion_rate = (
            conversions / clicks
            if clicks > 0 else 0
        )

        lead_conversion_rate = (
            conversions / leads
            if leads > 0 else 0
        )

        cost_per_click = (
            acquisition_cost / clicks
            if clicks > 0 else 0
        )

        input_df = pd.DataFrame({

            "duration": [duration],
            "impressions": [impressions],
            "clicks": [clicks],
            "leads": [leads],
            "conversions": [conversions],
            "acquisition_cost": [acquisition_cost],
            "engagement_score": [engagement_score],
            "total_acquisition_cost": [total_acquisition_cost],
            "click_through_rate": [click_through_rate],
            "conversion_rate": [conversion_rate],
            "lead_conversion_rate": [lead_conversion_rate],
            "cost_per_click": [cost_per_click],
            })

#ENCODING     
        campaign_columns = [
            "Email",
            "Influencer",
            "Paid Ads",
            "SEO",
            "Social Media"
        ]


        for col in campaign_columns:
            input_df[f"campaign_type_{col}"] = 0


        input_df[
            f"campaign_type_{campaign_type}"
        ] = 1



        audience_columns = [
            "College Students",
            "Premium Shoppers",
            "Tier 2 City Customers",
            "Working Women",
            "Youth"
        ]


        for col in audience_columns:
            input_df[f"target_audience_{col}"] = 0


        input_df[
            f"target_audience_{target_audience}"
        ] = 1



        brand_columns = [
            "nykaa",
            "purplle",
            "tira"
        ]


        for col in brand_columns:
            input_df[f"brand_{col}"] = 0


        input_df[
            f"brand_{brand}"
        ] = 1



        for col in audience_columns:
            input_df[
                f"customer_segment_{col}"
            ] = 0


        input_df[
            f"customer_segment_{target_audience}"
        ] = 1



        language_columns = [
            "Bengali",
            "English",
            "Hindi",
            "Tamil"
        ]


        for col in language_columns:
            input_df[f"language_{col}"] = 0


        input_df[
            f"language_{language}"
        ] = 1



        input_df = input_df.reindex(
            columns=regression_features,
            fill_value=0
        )

        

        prediction = regression_model.predict(input_df)

        profit_percentage = (((prediction[0] - total_acquisition_cost) / total_acquisition_cost) if total_acquisition_cost > 0 else 0)
        

        
        st.success(f"Predicted Revenue: ₹ {prediction[0]:,.2f}\n\nPredicted Profit Percentage: {profit_percentage:.2f} %")

        average_sale_value = (prediction[0] / conversions) if conversions > 0 else 0

        st.subheader("Calculated Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Engagement Score", engagement_score)
            st.metric("Click Through Rate", round(click_through_rate, 4))
        
        with col2:
            st.metric("Conversion Rate", round(conversion_rate, 4))
            st.metric("Lead Conversion Rate", round(lead_conversion_rate, 4))
        
        with col3:
            st.metric("Average Sale Value", "₹" + str(round(average_sale_value, 2)))
            st.metric("Total Acquisition Cost", "₹" + str(round(total_acquisition_cost, 2)))


        st.subheader("Campaign Performance Analysis")

        chart_df = pd.DataFrame({
        "Metric": ["Revenue", "Total Cost"],
        "Amount": [prediction[0], total_acquisition_cost]})

        st.bar_chart(chart_df.set_index("Metric"))

        funnel = pd.DataFrame({
        "Stage": [
            "Impressions",
            "Clicks",
            "Leads",
            "Conversions"
        ],
        "Count": [
            impressions,
            clicks,
            leads,
            conversions
        ]
        })

        st.subheader("Marketing Funnel")
        st.bar_chart(funnel.set_index("Stage"))


        
        



# PROFIT PREDICTION
elif page == "Profit Prediction":

    st.title("Profit / Loss Prediction")

    st.write(
        "Enter campaign details to predict whether the campaign will be profitable or not."
    )


    col1, col2 = st.columns(2)


    with col1:

        duration = st.number_input(
            "Duration in Days",
            min_value=1,
            value=30
        )

        impressions = st.number_input(
            "Impressions",
            min_value=0,
            value=50000
        )

        clicks = st.number_input(
            "Clicks",
            min_value=0,
            value=3000
        )

        leads = st.number_input(
            "Leads",
            min_value=0,
            value=1500
        )

        conversions = st.number_input(
            "Conversions",
            min_value=0,
            value=800
        )


    with col2:

        acquisition_cost = st.number_input(
            "Acquisition Cost",
            min_value=0.0,
            value=150.0
        )


        campaign_type = st.selectbox(
            "Campaign Type",
            [
                "Email",
                "Influencer",
                "Paid Ads",
                "SEO",
                "Social Media"
            ]
        )


        target_audience = st.selectbox(
            "Target Audience",
            [
                "College Students",
                "Premium Shoppers",
                "Tier 2 City Customers",
                "Working Women",
                "Youth"
            ]
        )


        brand = st.selectbox(
            "Brand",
            [
                "nykaa",
                "purplle",
                "tira"
            ]
        )


        language = st.selectbox(
            "Language",
            [
                "Bengali",
                "English",
                "Hindi",
                "Tamil"
            ]
        )


    predict_profit = st.button(
        "Predict Profit/Loss"
    )


    if predict_profit:



        engagement_score = (
            ((clicks + leads + conversions) / impressions) * 100
            if impressions > 0 else 0
        )


        click_through_rate = (
            clicks / impressions
            if impressions > 0 else 0
        )


        conversion_rate = (
            conversions / clicks
            if clicks > 0 else 0
        )


        lead_conversion_rate = (
            conversions / leads
            if leads > 0 else 0
        )


        cost_per_click = (
            acquisition_cost / clicks
            if clicks > 0 else 0
        )


        total_acquisition_cost = (
            acquisition_cost * conversions
        )




        classification_input = pd.DataFrame({

            "duration": [duration],
            "impressions": [impressions],
            "clicks": [clicks],
            "leads": [leads],
            "conversions": [conversions],
            "acquisition_cost": [acquisition_cost],
            "engagement_score": [engagement_score],
            "total_acquisition_cost": [total_acquisition_cost],
            "year": [2026],
            "month": [6],
            "day": [15],
            "click_through_rate": [click_through_rate],
            "conversion_rate": [conversion_rate],
            "lead_conversion_rate": [lead_conversion_rate],
            "cost_per_click": [cost_per_click],
                    })


        categorical_columns = [

            "campaign_type_Email",
            "campaign_type_Influencer",
            "campaign_type_Paid Ads",
            "campaign_type_SEO",
            "campaign_type_Social Media",
            "target_audience_College Students",
            "target_audience_Premium Shoppers",
            "target_audience_Tier 2 City Customers",
            "target_audience_Working Women",
            "target_audience_Youth",
            "brand_nykaa",
            "brand_purplle",
            "brand_tira",
            "customer_segment_College Students",
            "customer_segment_Premium Shoppers",
            "customer_segment_Tier 2 City Customers",
            "customer_segment_Working Women",
            "customer_segment_Youth",
            "language_Bengali",
            "language_English",
            "language_Hindi",
            "language_Tamil"
        ]


        for col in categorical_columns:
            classification_input[col] = 0


        classification_input[
            f"campaign_type_{campaign_type}"
        ] = 1


        classification_input[
            f"target_audience_{target_audience}"
        ] = 1


        classification_input[
            f"brand_{brand}"
        ] = 1


        classification_input[
            f"customer_segment_{target_audience}"
        ] = 1


        classification_input[
            f"language_{language}"
        ] = 1


        classification_input = classification_input.reindex(
            columns=classification_scaler.feature_names_in_,
            fill_value=0
        )


        #SCALE

        classification_input_scaled = classification_scaler.transform(
            classification_input
        )


        prediction = classification_model.predict(
            classification_input_scaled
        )


        result = label_encoder.inverse_transform(
            prediction
        )[0]

        regression_input = classification_input.copy()

        regression_input = regression_input.drop(columns=[
            "total_acquisition_cost",
            "year",
            "month",
            "day",
            "acquisition_cost_log",
            "customer_segment_College Students",
            "customer_segment_Premium Shoppers",
            "customer_segment_Tier 2 City Customers",
            "customer_segment_Working Women",
            "customer_segment_Youth"
            ], errors="ignore"
            )

        regression_input = regression_input.reindex(
            columns=regression_features,
            fill_value=0
        )



        revenue_prediction = regression_model.predict(regression_input)

        predicted_revenue = revenue_prediction[0]

        profit_amount = (predicted_revenue - total_acquisition_cost)

        roi = (profit_amount / total_acquisition_cost if total_acquisition_cost > 0 else 0 )

        st.success(f"""Predicted Revenue: ₹ {predicted_revenue:,.2f}\n\nProfit Amount: ₹ {profit_amount:,.2f}\n\nROI: {roi:.2f} %""")

        if roi > 0:
            final_result = "Profit"
        else:
            final_result = "Loss"


        if final_result == "Profit":

            st.markdown(
        """
        <h1 style="text-align:center;">
        ✅ PROFIT
        </h1>
        """,
        unsafe_allow_html=True
    )

        else:

            st.markdown(
                """
                <h1 style="text-align:center;">
                ❌ LOSS
                </h1>
                """,
                unsafe_allow_html=True
            )
