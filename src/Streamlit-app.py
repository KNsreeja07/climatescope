# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Analyse import (
    load_data, descriptive_stats, country_aggregates, correlation_matrix,
    monthly_trends, detect_extreme_events, region_comparisons
)

# st.set_page_config(page_title="🌦️ ClimateScope — Advanced Weather Dashboard", layout="wide")
# --- Page Configuration ---
st.set_page_config(
    page_title=" ClimateScope — Global Weather Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Styling ---
st.markdown("""
<style>
/* General app font and background */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    font-size: 20px !important
    background-color: #f5f6fa;
}
 .stMarkdown h2, .stMarkdown h3, .stSubheader {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #f5f6fa !important;
    }
            
/* Bold titles within write() */
strong, b {
        font-size: 20px !important;
        color: #f5f6fa !important;
    }          
# /* Title and headers */
# h1 {
#     color: #f5f6fa;
#     font-size: 38px !important;
#     font-weight: 700 !important;
# }
# h2, h3 {
#     color: #f5f6fa !important;
# }

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #334155;
    color: #f5f6fa;
}
.sidebar-content {
    padding: 1rem;
}

/* Metric boxes */
[data-testid="stMetricValue"] {
    color: #f5f6fa !important;
    font-weight: 600 !important;
}

/* Graph area */
.plot-container {
    background: white;
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 30px;
}
 /* Table text */
    .dataframe {
        font-size: 17px !important;
    }

    /* Axis and title text for plots */
    .js-plotly-plot .plotly .main-svg text {
        font-size: 16px !important;
    }
/* Links and hover */
a {
    color: #f5f6fa !important;
}
a:hover {
    text-decoration: underline !important;
}

/* Section separators */
hr {
    border: 1px solid #bf62d9;
    margin: 2rem 0;
}
            
            
</style>
""", unsafe_allow_html=True)


DATA_PATH = "../data/GlobalWeatherRepository_cleaned.csv"

@st.cache_data
def get_data():
    return load_data(DATA_PATH)

df = get_data()


with st.sidebar:
    st.markdown("## ClimateScope Dashboard")
    

    # --- Filters Section ---
    st.header("Filters")

    # Country Filter
    countries = sorted(df["country"].dropna().unique().tolist())
    selected_country = st.multiselect(
        "🌍 Country",
        countries,
        default=countries[:3],
        help="Select one or more countries to filter the data."
    )

    # Filter the dataframe dynamically
    dff = df[df["country"].isin(selected_country)] if selected_country else df

    # Date Range Filter
    date_min, date_max = df["date"].min(), df["date"].max()
    selected_dates = st.date_input(
        "📅 Date Range",
        (date_min, date_max)
    )

    # Metric Group Filter
    metric_group = st.selectbox(
        "📈 Metric Group",
        [
            "Temperature",
            "Humidity & Visibility",
            "Wind",
            "Precipitation & Pressure",
            "Air Quality",
            "Regional / Geographical",
            "Extreme Events"
        ],
        help="Select which climate category to analyze."
    )

    # --- Dynamic Data Overview ---
    st.markdown("### Data Overview")
    st.info(f"**Countries selected:** {len(selected_country)} / {len(countries)}")
    st.metric("Records in Selection", f"{dff.shape[0]:,}")





# --- Apply Filters ---
dff = df.copy()
if selected_country:
    dff = dff[dff["country"].isin(selected_country)]
if selected_dates and isinstance(selected_dates, tuple):
    start, end = selected_dates
    dff = dff[(dff["date"] >= pd.to_datetime(start)) & (dff["date"] <= pd.to_datetime(end))]


# --- Dashboard Header ---
st.title("ClimateScope — Global Weather Insights")

# --- Analysis Section ---
st.markdown("### 🔍 Analysis")

# --- Dynamic Analysis Links ---
if metric_group == "Temperature":
    st.markdown("""
    <div style='font-size:25px; line-height:1.8;'>
        <ul>
            <li><a href="#temp_trend" style="text-decoration:none;">📈 Temperature Trend Over Time</a></li>
            <li><a href="#temp_distribution" style="text-decoration:none;">📊 Temperature Distribution</a></li>
            <li><a href="#monthly_box" style="text-decoration:none;">📦 Monthly Temperature Distribution</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif metric_group == "Humidity & Visibility":
    st.markdown("""
    <div style='font-size:25px; line-height:1.8;'>
        <ul>
            <li><a href="#humid_temp" style="text-decoration:none;">💧 Humidity vs Temperature</a></li>
            <li><a href="#visibility_trend" style="text-decoration:none;">🌫️ Visibility Trend Over Time</a></li>
            <li><a href="#humid_corr" style="text-decoration:none;">🔗 Correlation Heatmap</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif metric_group == "Wind":
    st.markdown("""
    <div style='font-size:25px; line-height:1.8;'>
        <ul>
            <li><a href="#wind_trend" style="text-decoration:none;">🌬️ Wind Speed Trend</a></li>
            <li><a href="#gust_precip" style="text-decoration:none;">🌪️ Gusts vs Precipitation</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif metric_group == "Precipitation & Pressure":
    st.markdown("""
    <div style='font-size:25px; line-height:1.8;'>
        <ul>
            <li><a href="#precip_monthly" style="text-decoration:none;">🌧️ Average Monthly Precipitation</a></li>
            <li><a href="#pressure_trend" style="text-decoration:none;">📉 Pressure Trend Over Time</a></li>
            <li><a href="#pressure_precip" style="text-decoration:none;">💨 Pressure vs Precipitation</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif metric_group == "Air Quality":
    st.markdown("""
    <div style='font-size:25px; line-height:1.8;'>
        <ul>
            <li><a href="#aq_trend" style="text-decoration:none;">🌫️ Air Quality Trend Over Time</a></li>
            <li><a href="#aq_corr" style="text-decoration:none;">🔗 Pollutant Correlation Heatmap</a></li>
            <li><a href="#aq_region" style="text-decoration:none;">🗺️ Regional AQI Comparison</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif metric_group == "Regional / Geographical":
     st.markdown("""
    <div style='font-size:25px; line-height:1.8;'>
        <ul>
            <li><a href="#country_summary" style="text-decoration:none;">📋 Country-Level Climate Summary</a></li>
            <li><a href="#metric_map" style="text-decoration:none;">🗺️ Metric Map Visualization</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
     


elif metric_group == "Extreme Events":
    st.markdown("""
    <div style='font-size:25px; line-height:1.8;'>
        <ul>
            <li><a href="#thresholds" style="text-decoration:none;">⚙️ Thresholds Used</a></li>
            <li><a href="#extreme_table" style="text-decoration:none;">⚠️ Extreme Events Table</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- Metric Group Views ---

if metric_group == "Temperature":
    st.subheader("🌡️ Temperature Metrics")
    cols = ["temperature_celsius", "feels_like_celsius"]
    available = [c for c in cols if c in dff.columns]
    if not available:
        st.warning("Temperature columns not found.")
    else:
        # --- Temperature Trend ---
        st.markdown('<a name="temp_trend"></a>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        for c in available:
            ax.plot(dff["date"], dff[c], label=c, linewidth=2)
        ax.legend(fontsize=12)
        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("Temperature (°C)", fontsize=14)
        ax.set_title("Temperature Trend Over Time", fontsize=15, fontweight='bold')
        st.pyplot(fig)

        # --- Temperature Distribution ---
        st.markdown('<a name="temp_distribution"></a>', unsafe_allow_html=True)
       
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.histplot(dff["temperature_celsius"], kde=True, ax=ax2, color="coral")
        ax2.set_xlabel("Temperature (°C)", fontsize=13)
        ax2.set_ylabel("Frequency", fontsize=13)
        ax2.set_title("Distribution of Temperature", fontsize=15, fontweight='bold')
        st.pyplot(fig2)

        # --- Monthly Temperature Boxplot ---
        st.markdown('<a name="monthly_box"></a>', unsafe_allow_html=True)
        dff["month"] = dff["date"].dt.month
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.boxplot(
            x="month",
            y="temperature_celsius",
            hue="month",
            data=dff,
            palette="coolwarm",
            legend=False,
            ax=ax3
        )
        ax3.set_title("Monthly Temperature Distribution", fontsize=15, fontweight='bold')
        ax3.set_xlabel("Month", fontsize=13)
        ax3.set_ylabel("Temperature (°C)", fontsize=13)
        st.pyplot(fig3)

# if metric_group == "Temperature":
#     st.subheader("🌡️ Temperature Metrics")
#     cols = ["temperature_celsius", "feels_like_celsius"]
#     available = [c for c in cols if c in dff.columns]
#     if not available:
#         st.warning("Temperature columns not found.")
#     else:
#         # --- Temperature Trend ---
#         st.markdown('<a name="temp_trend"></a>', unsafe_allow_html=True)
#         st.write("**Temperature Trend Over Time**")
#         fig, ax = plt.subplots(figsize=(10,4))
#         for c in available:
#             ax.plot(dff["date"], dff[c], label=c)
#         ax.legend(); ax.set_xlabel("Date"); ax.set_ylabel("°C")
#         st.pyplot(fig)

#         # --- Temperature Distribution ---
#         st.markdown('<a name="temp_distribution"></a>', unsafe_allow_html=True)
#         st.write("**Temperature Distribution**")
#         fig2, ax2 = plt.subplots(figsize=(8,4))
#         sns.histplot(dff["temperature_celsius"], kde=True, ax=ax2, color="coral")
#         st.pyplot(fig2)

#         # --- Monthly Temperature Boxplot ---
#         st.markdown('<a name="monthly_box"></a>', unsafe_allow_html=True)
#         dff["month"] = dff["date"].dt.month
#         fig3, ax3 = plt.subplots(figsize=(8,4))
#         sns.boxplot(
#             x="month",
#             y="temperature_celsius",
#             hue="month",
#             data=dff,
#             palette="coolwarm",
#             legend=False,
#             ax=ax3
#         )
#         ax3.set_title("Monthly Temperature Distribution")
#         st.pyplot(fig3)

elif metric_group == "Humidity & Visibility":
    st.subheader("💧 Humidity & Visibility")

    if {"humidity","visibility_km","temperature_celsius"}.issubset(dff.columns):
        # --- Humidity vs Temperature ---
        st.markdown('<a name="humid_temp"></a>', unsafe_allow_html=True)
        # st.write("**Humidity vs Temperature**")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.set_title("Humidity Vs Temperature", fontsize=15, fontweight='bold')
        sns.scatterplot(data=dff, x="humidity", y="temperature_celsius", alpha=0.5, color="teal")
        st.pyplot(fig)

        # --- Visibility Trend ---
        st.markdown('<a name="visibility_trend"></a>', unsafe_allow_html=True)
        # st.write("**Visibility Trend Over Time**")
        fig2, ax2 = plt.subplots(figsize=(8,4))
        ax2.plot(dff["date"], dff["visibility_km"], color="orange")
        ax2.set_ylabel("Visibility (km)")
        ax2.set_title("Visibility Trend Over Time", fontsize=15, fontweight='bold')
        st.pyplot(fig2)

        # --- Correlation Heatmap ---
        st.markdown('<a name="humid_corr"></a>', unsafe_allow_html=True)
        # st.write("**Humidity–Visibility–Temperature Correlation Heatmap**")
        fig3, ax3 = plt.subplots(figsize=(6,4))
        ax3.set_title("Humidity–Visibility–Temperature Correlation Heatmap", fontsize=15, fontweight='bold')
        corr = dff[["humidity","visibility_km","temperature_celsius"]].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)

elif metric_group == "Wind":
    st.subheader("🌬️ Wind Metrics")

    if "wind_kph" in dff.columns:
        # --- Wind Speed Trend ---
        st.markdown('<a name="wind_trend"></a>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(dff["date"], dff["wind_kph"], label="wind_kph", color="blue")
        if "gust_kph" in dff.columns:
            ax.plot(dff["date"], dff["gust_kph"], label="gust_kph", color="red", alpha=0.6)
        ax.legend()
        ax.set_title("Wind Speed Trend")
        st.pyplot(fig)

        # --- Gust vs Precipitation ---
        st.markdown('<a name="gust_precip"></a>', unsafe_allow_html=True)
        if "gust_kph" in dff.columns and "precip_mm" in dff.columns:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            sns.scatterplot(data=dff, x="gust_kph", y="precip_mm", color="purple", ax=ax2)
            ax2.set_title("Gusts vs Precipitation")
            st.pyplot(fig2)
    else:
        st.warning("Wind data unavailable.")

elif metric_group == "Precipitation & Pressure":
    st.subheader("🌧️ Precipitation & Pressure")

    if "date" in dff.columns:
        dff["month"] = dff["date"].dt.to_period("M")
        monthly = dff.groupby("month")[["precip_mm","pressure_mb"]].mean().reset_index()

        # --- Average Monthly Precipitation ---
        st.markdown('<a name="precip_monthly"></a>', unsafe_allow_html=True)
        if "precip_mm" in monthly.columns:
            fig, ax = plt.subplots(figsize=(8,4))
            sns.barplot(x=monthly["month"].astype(str), y="precip_mm", data=monthly, color="skyblue", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_title("Average Monthly Precipitation")
            st.pyplot(fig)

        # --- Pressure Trend ---
        st.markdown('<a name="pressure_trend"></a>', unsafe_allow_html=True)
        if "pressure_mb" in dff.columns:
            fig2, ax2 = plt.subplots(figsize=(10,4))
            ax2.plot(dff["date"], dff["pressure_mb"], color="brown")
            ax2.set_ylabel("Pressure (mb)")
            ax2.set_title("Pressure Trend Over Time")
            st.pyplot(fig2)

        # --- Pressure vs Precipitation ---
        st.markdown('<a name="pressure_precip"></a>', unsafe_allow_html=True)
        if "pressure_mb" in dff.columns and "precip_mm" in dff.columns:
            fig3, ax3 = plt.subplots(figsize=(6,4))
            sns.scatterplot(data=dff, x="pressure_mb", y="precip_mm", color="gray", ax=ax3)
            ax3.set_title("Pressure vs Precipitation")
            st.pyplot(fig3)
    else:
        st.warning("Date column missing for monthly aggregation.")

elif metric_group == "Air Quality":
    st.subheader("🌫️ Air Quality Metrics")

    aq_cols = [c for c in ["air_quality_PM2.5","air_quality_PM10","air_quality_us-epa-index"] if c in dff.columns]
    if aq_cols:
        # --- AQ Trend ---
        st.markdown('<a name="aq_trend"></a>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10,4))
        for c in aq_cols:
            ax.plot(dff["date"], dff[c], label=c)
        ax.legend()
        ax.set_title("Air Quality Trend Over Time")
        st.pyplot(fig)

        # --- AQ Correlation ---
        st.markdown('<a name="aq_corr"></a>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5,4))
        sns.heatmap(dff[aq_cols].corr(), annot=True, cmap="YlOrRd", ax=ax2)
        ax2.set_title("Pollutant Correlation Heatmap")
        st.pyplot(fig2)

        # --- AQ Regional ---
        st.markdown('<a name="aq_region"></a>', unsafe_allow_html=True)
        if "country" in dff.columns:
            agg = dff.groupby("country")["air_quality_us-epa-index"].mean().sort_values(ascending=False).head(20)
            st.bar_chart(agg)
    else:
        st.warning("Air quality columns unavailable.")

elif metric_group == "Regional / Geographical":
    import plotly.express as px

    st.subheader("🗺️ Regional & Geographical Analysis")
    st.markdown('<a name="country_summary"></a>', unsafe_allow_html=True)
    # Compute country-level aggregates
    agg = country_aggregates(dff)
    if not agg.empty:
        st.write("**Country-Level Climate Summary**")
        st.dataframe(agg)

        # Pick a representative metric for map coloring
        st.markdown('<a name="metric_map"></a>', unsafe_allow_html=True)
        st.markdown(
    "<h4 style='font-size:25px; font-weight:600; color:#f5f6fa; margin-bottom:5px;'>Select Metric for Map Color</h4>",
    unsafe_allow_html=True
)
        metric_choice = st.selectbox(
            "",
            ["temperature_celsius", "humidity", "air_quality_us-epa-index", "pressure_mb", "precip_mm"]
        )

        if metric_choice in dff.columns:
            # Country-level mean metric
            country_means = (
                dff.groupby("country")[metric_choice]
                .mean()
                .reset_index()
                .sort_values(metric_choice, ascending=False)
            )

            fig_choro = px.choropleth(
                country_means,
                locations="country",
                locationmode="country names",
                color=metric_choice,
                color_continuous_scale="RdYlBu_r",
                title=f"Average {metric_choice.replace('_', ' ').title()} by Country",
                projection="natural earth",
                height=650,
            )
            fig_choro.update_geos(showcountries=True, showframe=True)

            st.plotly_chart(fig_choro, use_container_width=True)

            # Identify highest and lowest countries
            highest_country = country_means.iloc[0]
            lowest_country = country_means.iloc[-1]

            st.markdown(
                f"""
                <div style="font-size:20px; color:#dfe6e9; margin-top:15px;">
                    🌍 <b>Highest {metric_choice.replace('_', ' ').title()}:</b> 
                    <span style="color:#00cec9;">{highest_country['country']}</span> 
                    ({highest_country[metric_choice]:.2f})<br>
                    🧭 <b>Lowest {metric_choice.replace('_', ' ').title()}:</b> 
                    <span style="color:#fab1a0;">{lowest_country['country']}</span> 
                    ({lowest_country[metric_choice]:.2f})
                </div>
                """,
                unsafe_allow_html=True
            )
            
        else:
            st.warning(f"Column '{metric_choice}' not found in dataset.")
    else:
        st.warning("No country data available for aggregation.")

elif metric_group == "Extreme Events":
    st.subheader("⚠️ Extreme Weather Events")

    thresholds, ext_events, summary = detect_extreme_events(dff)

    # --- Thresholds Section ---
    st.markdown('<a name="thresholds"></a>', unsafe_allow_html=True)
    st.write("### Thresholds used to flag extremes:")
    st.json(thresholds)

    if not ext_events.empty:
        # --- Extreme Table ---
        st.markdown('<a name="extreme_table"></a>', unsafe_allow_html=True)
        display_cols = ["date", "country", "location_name", "temperature_celsius",
                        "wind_kph", "precip_mm", "visibility_km", "extreme_types"]
        df_display = ext_events[display_cols].copy()

        color_map = {
            "High Temp": "#FF6B6B",
            "High Wind": "#4D96FF",
            "Heavy Precip": "#4CAF50",
            "Low Visibility": ""
        }

        legend_html = """
        <b>Legend (Row Color):</b> 
        <span style='background-color:#FF6B6B;padding:2px 8px;margin-right:5px;'>High Temp</span>
        <span style='background-color:#4D96FF;padding:2px 8px;margin-right:5px;'>High Wind</span>
        <span style='background-color:#4CAF50;padding:2px 8px;margin-right:5px;'>Heavy Precip</span>
        <span style='background-color:transparent;padding:2px 8px;margin-right:5px;'>Low Visibility</span>
        """
        st.markdown(legend_html, unsafe_allow_html=True)

        def highlight_extremes(row):
            if row["extreme_types"]:
                for key in color_map:
                    if key.lower() in row["extreme_types"].lower() and color_map[key]:
                        return ['background-color: ' + color_map[key]] * len(row)
            return [''] * len(row)

        st.write("### Extreme Events Table (highlighted by type)")
        st.dataframe(df_display.style.apply(highlight_extremes, axis=1))
    else:
        st.info("No extreme events detected in this selection.")

# --- Footer ---
st.markdown("---")

# # app_streamlit.py (Enhanced UI Edition)
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# from Analyse import (
#     load_data, descriptive_stats, country_aggregates, correlation_matrix,
#     monthly_trends, detect_extreme_events, region_comparisons
# )

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="🌦️ ClimateScope — Global Weather Insights",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- Global Styling ---
# st.markdown("""
# <style>
# html, body, [class*="css"] {
#     font-family: 'Poppins', sans-serif;
#     background-color: #f7f9fb;
# }

# /* Title and headers */
# h1 {
#     color: #1E3A8A !important;
#     font-size: 38px !important;
#     font-weight: 700 !important;
# }
# h2, h3 {
#     color: #334155 !important;
#     font-weight: 600 !important;
# }

# /* Sidebar */
# section[data-testid="stSidebar"] {
#     background-color: #e0e7ff;
#     color: #1e293b;
#     border-right: 1px solid #c7d2fe;
# }

# /* Metric and Info */
# [data-testid="stMetricValue"] {
#     color: #1E40AF !important;
#     font-weight: 600 !important;
# }
# [data-testid="stMetricLabel"] {
#     color: #334155 !important;
# }

# /* Plot Containers */
# .plot-container {
#     background: white;
#     border-radius: 12px;
#     padding: 16px;
#     box-shadow: 0 2px 8px rgba(0,0,0,0.05);
#     margin-bottom: 32px;
# }

# /* Links */
# a {
#     color: #1E3A8A !important;
#     text-decoration: none;
# }
# a:hover {
#     text-decoration: underline !important;
# }

# /* Footer */
# hr {
#     border: 1px solid #CBD5E1;
#     margin: 2rem 0;
# }
# </style>
# """, unsafe_allow_html=True)

# # --- Load Data ---
# DATA_PATH = "../data/GlobalWeatherRepository_cleaned.csv"

# @st.cache_data
# def get_data():
#     return load_data(DATA_PATH)

# df = get_data()

# # --- Sidebar ---
# with st.sidebar:
#     st.markdown("### 🌍 ClimateScope Dashboard")
#     st.markdown("---")

#     # --- Filters ---
#     st.header("Filters")

#     countries = sorted(df["country"].dropna().unique().tolist())
#     selected_country = st.multiselect(
#         "Select Countries",
#         countries,
#         default=countries[:3]
#     )

#     dff = df[df["country"].isin(selected_country)] if selected_country else df

#     date_min, date_max = df["date"].min(), df["date"].max()
#     selected_dates = st.date_input("Date Range", (date_min, date_max))

#     metric_group = st.selectbox(
#         "Metric Group",
#         [
#             "Temperature",
#             "Humidity & Visibility",
#             "Wind",
#             "Precipitation & Pressure",
#             "Air Quality",
#             "Regional / Geographical",
#             "Extreme Events"
#         ]
#     )

#     st.markdown("---")
#     st.markdown("#### 📊 Data Overview")
#     st.info(f"**Countries selected:** {len(selected_country)} / {len(countries)}")
#     st.metric("Records in Selection", f"{dff.shape[0]:,}")

# # --- Apply Filters ---
# if selected_country:
#     dff = dff[dff["country"].isin(selected_country)]
# if selected_dates and isinstance(selected_dates, tuple):
#     start, end = selected_dates
#     dff = dff[(dff["date"] >= pd.to_datetime(start)) & (dff["date"] <= pd.to_datetime(end))]

# # --- Header ---
# st.title("🌦️ ClimateScope — Global Weather Insights")
# st.markdown("### 🔍 Interactive Climate Analytics")
# st.markdown("---")

# # --- Unified Figure Style ---
# plt.rcParams.update({
#     "figure.figsize": (10, 5),
#     "axes.titlesize": 13,
#     "axes.labelsize": 11,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10
# })

# # Utility function to wrap plots with consistent card style
# def plot_card(title, fig):
#     st.markdown('<div class="plot-container">', unsafe_allow_html=True)
#     st.write(f"**{title}**")
#     st.pyplot(fig)
#     st.markdown('</div>', unsafe_allow_html=True)

# # --- Metric Group Visualizations ---
# if metric_group == "Temperature":
#     st.subheader("🌡️ Temperature Metrics")
#     cols = ["temperature_celsius", "feels_like_celsius"]
#     available = [c for c in cols if c in dff.columns]

#     if available:
#         fig, ax = plt.subplots()
#         for c in available:
#             ax.plot(dff["date"], dff[c], label=c, linewidth=2)
#         ax.legend()
#         ax.set_xlabel("Date"); ax.set_ylabel("°C")
#         plot_card("Temperature Trend Over Time", fig)

#         fig2, ax2 = plt.subplots()
#         sns.histplot(dff["temperature_celsius"], kde=True, ax=ax2, color="#F97316")
#         plot_card("Temperature Distribution", fig2)

#         dff["month"] = dff["date"].dt.month
#         fig3, ax3 = plt.subplots()
#         sns.boxplot(x="month", y="temperature_celsius", data=dff, palette="coolwarm", ax=ax3)
#         ax3.set_title("Monthly Temperature Spread")
#         plot_card("Monthly Temperature Distribution", fig3)
#     else:
#         st.warning("Temperature data unavailable.")

# elif metric_group == "Humidity & Visibility":
#     st.subheader("💧 Humidity & Visibility")
#     if {"humidity","visibility_km","temperature_celsius"}.issubset(dff.columns):
#         fig, ax = plt.subplots()
#         sns.scatterplot(data=dff, x="humidity", y="temperature_celsius", alpha=0.5, color="teal", ax=ax)
#         plot_card("Humidity vs Temperature", fig)

#         fig2, ax2 = plt.subplots()
#         ax2.plot(dff["date"], dff["visibility_km"], color="orange")
#         ax2.set_ylabel("Visibility (km)")
#         plot_card("Visibility Trend Over Time", fig2)

#         fig3, ax3 = plt.subplots()
#         corr = dff[["humidity","visibility_km","temperature_celsius"]].corr()
#         sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
#         plot_card("Humidity–Visibility–Temperature Correlation Heatmap", fig3)

# elif metric_group == "Wind":
#     st.subheader("🌬️ Wind Metrics")
#     if "wind_kph" in dff.columns:
#         fig, ax = plt.subplots()
#         ax.plot(dff["date"], dff["wind_kph"], label="wind_kph", color="#2563EB", linewidth=2)
#         if "gust_kph" in dff.columns:
#             ax.plot(dff["date"], dff["gust_kph"], label="gust_kph", color="#F43F5E", alpha=0.7)
#         ax.legend()
#         plot_card("Wind Speed Trend", fig)

#         if "gust_kph" in dff.columns and "precip_mm" in dff.columns:
#             fig2, ax2 = plt.subplots()
#             sns.scatterplot(data=dff, x="gust_kph", y="precip_mm", color="#7C3AED", ax=ax2)
#             plot_card("Gusts vs Precipitation", fig2)
#     else:
#         st.warning("Wind data unavailable.")

# elif metric_group == "Precipitation & Pressure":
#     st.subheader("🌧️ Precipitation & Pressure")
#     if "date" in dff.columns:
#         dff["month"] = dff["date"].dt.to_period("M")
#         monthly = dff.groupby("month")[["precip_mm","pressure_mb"]].mean().reset_index()

#         if "precip_mm" in monthly.columns:
#             fig, ax = plt.subplots()
#             sns.barplot(x=monthly["month"].astype(str), y="precip_mm", data=monthly, color="#38BDF8", ax=ax)
#             ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
#             plot_card("Average Monthly Precipitation", fig)

#         if "pressure_mb" in dff.columns:
#             fig2, ax2 = plt.subplots()
#             ax2.plot(dff["date"], dff["pressure_mb"], color="#78350F")
#             ax2.set_ylabel("Pressure (mb)")
#             plot_card("Pressure Trend Over Time", fig2)

#         if "pressure_mb" in dff.columns and "precip_mm" in dff.columns:
#             fig3, ax3 = plt.subplots()
#             sns.scatterplot(data=dff, x="pressure_mb", y="precip_mm", color="#475569", ax=ax3)
#             plot_card("Pressure vs Precipitation", fig3)
#     else:
#         st.warning("Date column missing for monthly aggregation.")

# elif metric_group == "Air Quality":
#     st.subheader("🌫️ Air Quality Metrics")
#     aq_cols = [c for c in ["air_quality_PM2.5","air_quality_PM10","air_quality_us-epa-index"] if c in dff.columns]
#     if aq_cols:
#         fig, ax = plt.subplots()
#         for c in aq_cols:
#             ax.plot(dff["date"], dff[c], label=c)
#         ax.legend()
#         plot_card("Air Quality Trend Over Time", fig)

#         fig2, ax2 = plt.subplots()
#         sns.heatmap(dff[aq_cols].corr(), annot=True, cmap="YlOrRd", ax=ax2)
#         plot_card("Pollutant Correlation Heatmap", fig2)

#         if "country" in dff.columns:
#             agg = dff.groupby("country")["air_quality_us-epa-index"].mean().sort_values(ascending=False).head(20)
#             st.bar_chart(agg)
#     else:
#         st.warning("Air quality columns unavailable.")

# elif metric_group == "Regional / Geographical":
#     st.subheader("🗺️ Regional & Geographical Analysis")
#     agg = country_aggregates(dff)
#     if not agg.empty:
#         st.write("**Country-Level Climate Summary**")
#         st.dataframe(agg)

#         metric_choice = st.selectbox(
#             "Select metric for map color",
#             ["temperature_celsius", "humidity", "air_quality_us-epa-index", "pressure_mb", "precip_mm"]
#         )
#         if metric_choice in dff.columns:
#             country_means = (
#                 dff.groupby("country")[metric_choice]
#                 .mean()
#                 .reset_index()
#                 .sort_values(metric_choice, ascending=False)
#             )
#             fig_choro = px.choropleth(
#                 country_means,
#                 locations="country",
#                 locationmode="country names",
#                 color=metric_choice,
#                 color_continuous_scale="RdYlBu_r",
#                 title=f"Average {metric_choice} by Country",
#                 projection="natural earth",
#                 height=650,
#             )
#             fig_choro.update_geos(showcountries=True, showframe=True)
#             st.plotly_chart(fig_choro, use_container_width=True)
#         else:
#             st.warning(f"Column '{metric_choice}' not found.")
#     else:
#         st.warning("No country data available for aggregation.")

# elif metric_group == "Extreme Events":
#     st.subheader("⚠️ Extreme Weather Events")
#     thresholds, ext_events, summary = detect_extreme_events(dff)

#     st.write("### Thresholds used to flag extremes:")
#     st.json(thresholds)

#     if not ext_events.empty:
#         display_cols = ["date", "country", "location_name", "temperature_celsius",
#                         "wind_kph", "precip_mm", "visibility_km", "extreme_types"]
#         df_display = ext_events[display_cols].copy()

#         color_map = {
#             "High Temp": "#FF6B6B",
#             "High Wind": "#4D96FF",
#             "Heavy Precip": "#4CAF50",
#         }

#         def highlight_extremes(row):
#             if row["extreme_types"]:
#                 for key in color_map:
#                     if key.lower() in row["extreme_types"].lower():
#                         return ['background-color: ' + color_map[key]] * len(row)
#             return [''] * len(row)

#         st.write("### Extreme Events Table (highlighted by type)")
#         st.dataframe(df_display.style.apply(highlight_extremes, axis=1))
#     else:
#         st.info("No extreme events detected in this selection.")

# # --- Footer ---
# st.markdown("---")
# st.caption("© 2025 ClimateScope | Interactive Climate Data Visualization Dashboard 🌍")

