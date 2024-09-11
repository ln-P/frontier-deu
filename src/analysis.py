"""
Run analysis pipeline
"""

import numpy as np
import pandas as pd

from src.data_load import create_analysis_dataset, process_raw_data
from src.eda import run_eda
from src.models import run_modeling

if __name__ == "__main__":
    # TODO: replace print with log
    print("Running data load")
    process_raw_data()
    create_analysis_dataset()

    print("Running EDA")
    run_eda()

    print("Running modelling")
    deu = pd.read_csv("./data/processed/deu_analysis_data.csv")

    selected_features = [
        "estimated_population",
        "number_of_connected_electricity_meters",
        # 'electricity_consumption_kwh',
        # 'mean_electricity_consumption_kwh_per_meter',
        "number_of_consuming_gas_meters",
        # 'gas_consumption_kwh',
        # 'mean_gas_consumption_kwh_per_meter',
        # 'average_energy_consumption_per_person_kwh',
        "band_a",
        "band_b",
        "band_c",
        "band_d",
        "band_e",
        "band_f",
        "band_g",
        "rural_town_and_fringe_in_a_sparse_setting",
        "rural_village_and_dispersed",
        "rural_village_and_dispersed_in_a_sparse_setting",
        "urban_city_and_town",
        "urban_city_and_town_in_a_sparse_setting",
        "urban_major_conurbation",
        "urban_minor_conurbation",
        # 'co_2_emissions_current',
        # 'lighting_cost_current',
        # 'heating_cost_current',
        # 'hot_water_cost_current',
        "number_lodgements",
        "avg_co_2_emissions",
        "avg_lighting_cost",
        "avg_heating_cost_current",
        "avg_hot_water_cost",
        # '1r_0p',
        "1r_1p",
        "1r_2p",
        "1r_3p",
        "1r_4p",
        # '2r_0p',
        "2r_1p",
        "2r_2p",
        "2r_3p",
        "2r_4p",
        # '3r_0p',
        "3r_1p",
        "3r_2p",
        "3r_3p",
        "3r_4p",
        # '4r_0p',
        "4r_1p",
        "4r_2p",
        "4r_3p",
        "4r_4p",
        # '5r_0p',
        "5r_1p",
        "5r_2p",
        "5r_3p",
        "5r_4p",
        # '6r_0p',
        "6r_1p",
        "6r_2p",
        "6r_3p",
        "6r_4p",
        "tax_a",
        "tax_b",
        "tax_c",
        "tax_d",
        "tax_e",
        "tax_f",
        "tax_g",
        "tax_h",
        "tax_i",
        "is_wales",
    ]

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
        # 'rural_town_and_fringe_in_a_sparse_setting',
        # 'rural_village_and_dispersed', 'rural_village_and_dispersed_in_a_sparse_setting',
        # 'urban_city_and_town', 'urban_city_and_town_in_a_sparse_setting',
        # 'urban_major_conurbation', 'urban_minor_conurbation',
        "number_lodgements",
        "avg_co_2_emissions",
        "avg_lighting_cost",
        "avg_heating_cost_current",
        "avg_hot_water_cost",
        # '1r_0p',
        "1r_1p",
        "1r_2p",
        "1r_3p",
        "1r_4p",
        # '2r_0p',
        "2r_1p",
        "2r_2p",
        "2r_3p",
        "2r_4p",
        # '3r_0p',
        "3r_1p",
        "3r_2p",
        "3r_3p",
        "3r_4p",
        # '4r_0p',
        "4r_1p",
        "4r_2p",
        "4r_3p",
        "4r_4p",
        # '5r_0p',
        "5r_1p",
        "5r_2p",
        "5r_3p",
        "5r_4p",
        # '6r_0p',
        "6r_1p",
        "6r_2p",
        "6r_3p",
        "6r_4p",
        "tax_a",
        "tax_b",
        "tax_c",
        "tax_d",
        "tax_e",
        "tax_f",
        "tax_g",
        "tax_h",
        "tax_i",
        # 'is_wales'
    ]

    print("Creating baseline models")
    run_modeling(
        deu, selected_features, "total_energy_consumption_kwh", prefix="baseline_"
    )

    print("Creating log models")
    # Transform to log - log1p to handles zero values
    deu["log_total_energy_consumption_kwh"] = np.log1p(
        deu["total_energy_consumption_kwh"]
    )

    deu_log = deu[["log_total_energy_consumption_kwh"] + selected_features].copy()
    for feature in continuous_features:
        deu_log[feature] = np.log1p(deu[feature])

    run_modeling(
        deu_log, selected_features, "log_total_energy_consumption_kwh", prefix="log_"
    )
