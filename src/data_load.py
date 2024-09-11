"""
Script to load and pre-process all of the analysis data
"""

import pandas as pd

from src.utils import check_missing_values, check_unique_id, clean_header


def clean_energy_consumption(file_path: str) -> pd.DataFrame:
    energy_consumption = (
        pd.read_csv(file_path)
        .rename(columns=clean_header)
        .rename(columns={"lower_layer_super_output_area_lsoa_code": "lsoa_code"})
        .drop(
            columns=[
                "local_authority_name",
                "local_authority_code",
                "msoa_name",
                "middle_layer_super_output_area_msoa_code",
            ]
        )
    )
    return energy_consumption


def clean_housing_efficiency(file_path: str) -> pd.DataFrame:
    # Skipping header rows and footer for relevant data only
    housing_efficiency = (
        pd.read_csv(file_path, sep=",", skiprows=4, skipfooter=4, engine="python")
        .rename(columns=clean_header)
    )

    housing_efficiency[["geo", "lsoa_code", "lsoa_name"]] = housing_efficiency["area"].str.split(":", expand=True)

    # Remove England and Wales summary row
    housing_efficiency = housing_efficiency[housing_efficiency.geo == "lsoa2021"]

    housing_efficiency[["lsoa_code", "lsoa_name"]] = housing_efficiency[
        ["lsoa_code", "lsoa_name"]
    ].map(str.strip)
    to_keep = ["lsoa_code", "lsoa_name"] + [
        c for c in housing_efficiency.columns if c.startswith("band_")
    ]

    return housing_efficiency[to_keep]


def clean_household_size(file_path: str) -> pd.DataFrame:

    household_size = (
        pd.read_csv(file_path)
        .rename(columns=clean_header)
        .rename(
            columns={
                "lower_layer_super_output_areas_code": "lsoa_code",
                "lower_layer_super_output_areas": "lsoa_name",
            }
        )
        # Simplify category names
        .assign(
            number_of_rooms_categories=lambda col: col.number_of_rooms_valuation_office_agency_6_categories.str.replace(
                "( or more)? rooms?", "r", regex=True
            )
        )
        .assign(
            household_size_categories=lambda col: col.household_size_5_categories.str.replace(
                "( or more)? (people|person) in household", "p", regex=True
            )
        )
        # Pivot categories to wide, one LSOA per row
        .pivot_table(
            index=["lsoa_code", "lsoa_name"],
            columns=["number_of_rooms_categories", "household_size_categories"],
            values="observation",
            fill_value=0,
        )
    )

    household_size.columns = [
        "_".join(col).strip() for col in household_size.columns.values
    ]
    household_size = household_size.reset_index()

    return household_size


def clean_tax_bands(file_path: str) -> pd.DataFrame:
    # Dwellings by council tax band
    # In notes: "All counts are rounded to the nearest 10, with counts between 1 and 4 suppressed and presented as "-".". Still there are zeros in the data.
    tax_bands = (
        pd.read_excel(file_path, sheet_name="CTSOP1.1_2023", skiprows=4)
        .query("geography == 'LSOA'")
        .filter(regex="ecode|band*")
        .rename(
            columns=lambda c: c.replace("band_", "tax_") if c.startswith("band_") else c
        )
        .rename(columns={"ecode": "lsoa_code"})
        # Replace - with 1 as they should be non-zero
        .replace("-", 1)
        .replace("..", 0)
    )

    tax_columns = tax_bands.filter(regex="^tax_").columns
    tax_bands[tax_columns] = tax_bands[tax_columns].astype(int)
    return tax_bands


def clean_energy_costs(file_path: str) -> pd.DataFrame:
    # Domestic energy usage data
    domestic_energy_costs = (
        pd.read_excel(
            "./data/raw/D3-_Domestic_Properties.ods", sheet_name="D3_by_LA", skiprows=3
        )
        # Focus on 2021 for consistency with other datasets
        .query("Quarter.str.contains('2021')")
        .rename(columns=clean_header)
        # Impute yearly metrics
        .groupby(["local_authority_code", "region"])
        .agg(
            {
                "number_lodgements": "mean",
                "co_2_emissions_current": "sum",
                "lighting_cost_current": "sum",
                "heating_cost_current": "sum",
                "hot_water_cost_current": "sum",
            }
        )
        .reset_index()
        .assign(
            avg_co_2_emissions=lambda c: c.co_2_emissions_current / c.number_lodgements,
            avg_lighting_cost=lambda c: c.lighting_cost_current / c.number_lodgements,
            avg_heating_cost_current=lambda c: c.heating_cost_current
            / c.number_lodgements,
            avg_hot_water_cost=lambda c: c.hot_water_cost_current / c.number_lodgements,
        )
    )

    return domestic_energy_costs


def load_lsoa_lad_lookup(file_path: str) -> pd.DataFrame:
    # Mapping file between LSOA and LAD
    lsoa_lad = (
        pd.read_csv(file_path, encoding="latin1")
        .drop_duplicates(subset=["lsoa11cd", "ladcd"])
        .filter(["lsoa11nm", "lsoa11cd", "ladcd"])
        # We need only England or Wales
        .query("ladcd.str.contains('E|W', regex=True, na=False)")
        .rename(columns={"lsoa11cd": "lsoa_code", "ladcd": "local_authority_code"})
    )
    return lsoa_lad


def clean_urban_rural_indicator(file_path: str) -> pd.DataFrame:
    # Issue is that the rural urban is based on 2011 data
    urban_rural = (
        pd.read_csv(file_path)
        .rename(columns=clean_header)
        .rename(columns={"lsoa11cd": "lsoa_code", "ruc11": "rural_urban"})
        .assign(rural_urban=lambda df: df.rural_urban.apply(clean_header))
        .filter(['lsoa_code', 'ruc11cd', 'rural_urban'])
    )
    return urban_rural

def process_raw_data():
    # Load all datasets and check uniqe ID and missing values
    for d in DATA_DICTIONARY.keys():
        print(d)
        df = DATA_DICTIONARY[d]['func'](DATA_DICTIONARY[d]['raw_path'])
        check_unique_id(df, DATA_DICTIONARY[d]['id'])
        check_missing_values(df, f"./output/eda/missing_values_{d}.png")
        
        df.to_csv(f"./data/processed/{d}.csv", index=False)

def create_analysis_dataset():
    PROCESSED_PATH = "./data/processed/"

    energy_consumption = pd.read_csv(PROCESSED_PATH + "energy_consumption.csv") # 33811 rows, I think this is still based on 2011 census
    housing_efficiency = pd.read_csv(PROCESSED_PATH + "housing_efficiency.csv") # 35672 
    household_size = pd.read_csv(PROCESSED_PATH + "household_size.csv") # 35672 
    tax_bands = pd.read_csv(PROCESSED_PATH + "tax_bands.csv") # 35672 
    domestic_energy_costs = pd.read_csv(PROCESSED_PATH + "domestic_energy_costs.csv") # 339  rows as they are based on LOCAL Authority code 
    urban_rural = pd.read_csv(PROCESSED_PATH + "urban_rural.csv") # 34753  as it is 2011 census  
    lsoa_lad_lookup = pd.read_csv(PROCESSED_PATH + "lsoa_lad_lookup.csv") # 34753 as it is 2011 census  

    # 2021 census boundries data 
    deu_2021 = pd.merge(housing_efficiency, household_size, how='outer', validate='1:1')
    deu_2021 = pd.merge(deu_2021, tax_bands, how='outer', validate='1:1')

    # 2011 census boundries data 
    deu_2011 = pd.merge(lsoa_lad_lookup, domestic_energy_costs, how='left', validate='m:1') # all keys merged
    deu_2011 = pd.merge(deu_2011, urban_rural, how='inner', validate='1:1')

    # Merge consumption onto other datasets 
    deu = pd.merge(energy_consumption, deu_2021, how='inner')
    deu = pd.merge(deu, deu_2011, how='inner')

    # Remove the non_consuming_gas_meters from sample as this is the only column with missing, see EDA charts
    deu = deu.drop(columns=['numer_of_non_consuming_gas_meters'])
    deu = deu[~deu.isna().any(axis=1)]

    # Add two features
    deu['is_wales'] = deu['lsoa_code'].str.startswith('W') 
    deu = pd.get_dummies(deu, columns=['rural_urban'], prefix="", prefix_sep="", drop_first=True) # rural_town_and_fringe is base level 

    deu.to_csv(PROCESSED_PATH + "/deu_analysis_data.csv", index=False)


def load_analysis_dataset():
    return pd.read_csv("./data/processed/deu_analysis_data.csv")


DATA_DICTIONARY = {
    'energy_consumption': {
        'raw_path': "./data/raw/LSOA Energy Consumption Data.csv",
        'func': clean_energy_consumption,
        'id': 'lsoa_code'
    },
    'housing_efficiency': {
        'raw_path': "./data/raw/nomis_energy_efficiency_of_housing.csv",
        'func': clean_housing_efficiency,
        'id': 'lsoa_code'
    },
    'household_size': {
        'raw_path': "./data/raw/RM202-Household-Size-By-Number-Of-Rooms-2021-lsoa-ONS.csv",
        'func': clean_household_size,
        'id': 'lsoa_code'
    },
    'tax_bands': {
        'raw_path': "./data/raw/CTSOP1.1_2023.xlsx",
        'func': clean_tax_bands,
        'id': 'lsoa_code'
    },
    'domestic_energy_costs': {
        'raw_path': "./data/raw/D3-_Domestic_Properties.ods",
        'func': clean_energy_costs,
        'id': 'local_authority_code'
    },
    'lsoa_lad_lookup': {
        'raw_path': "./data/raw/PCD_OA_LSOA_MSOA_LAD_AUG22_UK_LU.csv",
        'func': load_lsoa_lad_lookup,
        'id': 'lsoa_code'
    },
    'urban_rural': {
        'raw_path': "./data/raw/Rural_Urban_Classification_(2011)_of_Lower_Layer_Super_Output_Areas_in_England_and_Wales.csv",
        'func': clean_urban_rural_indicator,
        'id': 'lsoa_code'
    }
}


if __name__ == "__main__":
    # Load raw -> processed
    process_raw_data()

    # Combine processed data
    create_analysis_dataset()
