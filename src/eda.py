import branca.colormap as cm
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_load import load_analysis_dataset

sns.set_style("ticks")
palette = sns.color_palette("Set2")

deu = load_analysis_dataset()

# Define relevant variables
non_features = [
    "total_energy_consumption_kwh",
    "lsoa_name",
    "lsoa_code",
    "latitude",
    "longitude",
    "shape_area",
    "lsoa11nm",
    "local_authority_code",
    "region",
    "ruc11cd",
]
all_features = [i for i in list(deu.columns) if i not in non_features]
continuous_features = [
    "estimated_population",
    "number_of_connected_electricity_meters",
    "number_of_consuming_gas_meters",
    "band_a",
    "band_b",
    "band_c",
    "band_d",
    "band_e",
    "band_f",
    "band_g",
    "number_lodgements",
    "avg_co_2_emissions",
    "avg_lighting_cost",
    "avg_heating_cost_current",
    "avg_hot_water_cost",
    "tax_a",
    "tax_b",
    "tax_c",
    "tax_d",
    "tax_e",
    "tax_f",
    "tax_g",
    "tax_h",
    "tax_i",
]
target = "total_energy_consumption_kwh"


def plot_correlation_matrix(data, features, target):
    # Create correlation matrix
    corr_matrix = data[features + [target]].corr()
    plt.figure(figsize=(20, 16))
    heatmap = sns.heatmap(corr_matrix, cmap="coolwarm", cbar=True)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
    plt.title("Correlation Matrix - All features")
    plt.tight_layout()
    plt.savefig("./output/eda/correlation_matrix_all_variables.png")


def plot_energy_distribution(data, target, palette):
    # Plot histogram of target variable
    plt.figure(figsize=(12, 8))
    sns.histplot(data[target], bins=50, kde=True, color=palette[2])
    plt.title("Distribution of Total Energy Consumption (kWh)")
    plt.xlabel("Total Energy Consumption (kWh)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("./output/eda/distribution_energy_consumption.png")


def plot_log_energy_distribution(data, target, palette):
    data["log_total_energy_consumption_kwh"] = np.log1p(data[target])
    plt.figure(figsize=(12, 8))
    sns.histplot(
        data["log_total_energy_consumption_kwh"], bins=50, kde=True, color=palette[2]
    )
    plt.title("Distribution of Log Total Energy Consumption (kWh)")
    plt.xlabel("Log(Total Energy Consumption (kWh))")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("./output/eda/distribution_log_energy_consumption.png")


def plot_pairplot(data, features, target):
    pairplot = sns.pairplot(data=data[features + [target]], kind="hist")
    pairplot.savefig(f"./output/eda/pairplot_analysis_data.png")


def create_folium_map(data, geojson_path):
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.merge(data, left_on="LSOA21CD", right_on="lsoa_code")

    m = folium.Map([53.978, -2.085], tiles="Cartodb Positron", zoom_start=7)

    # Reverse palette
    colormap = cm.linear.RdYlBu_10.scale(
        deu["average_energy_consumption_per_person_kwh"].min(),
        deu["average_energy_consumption_per_person_kwh"].max(),
    )
    colormap = colormap.to_step(n=10)
    colormap.colors.reverse()

    def style_function(feature, colormap, feature_name):
        lsoa_code = feature["properties"]["LSOA21CD"]
        value = data.loc[data["lsoa_code"] == lsoa_code, feature_name].values[0]
        return {
            "fillColor": colormap(value),
            "fillOpacity": 0.7,
            "weight": 0.2,
            "color": "grey",
        }

    def add_geojson_layer(
        gdf, colormap, feature_name, layer_name, tooltip_fields, tooltip_aliases
    ):
        folium.GeoJson(
            gdf,
            style_function=lambda feature: style_function(
                feature, colormap, feature_name
            ),
            name=layer_name,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields, aliases=tooltip_aliases, localize=True
            ),
        ).add_to(m)

    add_geojson_layer(
        gdf,
        colormap,
        "average_energy_consumption_per_person_kwh",
        "Energy Consumption",
        [
            "lsoa_name",
            "average_energy_consumption_per_person_kwh",
            "estimated_population",
        ],
        ["LSOA Name: ", "Average Energy Consumption (kWh): ", "Est. Population: "],
    )

    colormap.add_to(m)

    folium.LayerControl().add_to(m)
    m.save("./output/eda/energy_consumption_map.html")


def run_eda(pairplot=False):
    plot_correlation_matrix(deu, all_features, target)
    plot_energy_distribution(deu, target, palette)
    plot_log_energy_distribution(deu, target, palette)
    print("Creating energy consumption map")
    create_folium_map(
        deu,
        "./data/raw/Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4_-3499630105124417570.geojson",
    )
    if pairplot:
        print("Creating pairplot")
        plot_pairplot(deu, continuous_features, target)


if __name__ == "__main__":
    # TODO: replace print with logs
    print("Running EDA pipeline")
    run_eda()
