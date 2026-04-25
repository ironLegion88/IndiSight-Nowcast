import pandas as pd
import geopandas as gpd
import re
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(module_name=__name__, log_sub_dir="data_engine")

class DataIngestionPipeline:
    def __init__(self):
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.tabular_out = self.processed_dir / "tabular"
        self.spatial_out = self.processed_dir / "spatial"
        
        self.tabular_out.mkdir(parents=True, exist_ok=True)
        self.spatial_out.mkdir(parents=True, exist_ok=True)

        # Exhaustive explicit mapping to common variable names
        # Ordered from most specific to least specific to prevent substring collisions
        self.metric_mapping = {
            "severely_wasted": "child_severely_wasted",
            "are_wasted": "child_wasted",
            "stunted": "child_stunted",
            "underweight": "child_underweight",
            "overweight_weight_for_height": "child_overweight",
            
            "non_pregnant_women_age_group_15_to_49_years_who_are_anaemic": "women_non_pregnant_anaemic",
            "pregnant_women_age_group_15_to_49_years_who_are_anaemic": "women_pregnant_anaemic",
            "women_age_group_15_to_19_years_who_are_anaemic": "women_15_19_anaemic",
            "women_age_group_15_to_49_years_who_are_anaemic": "women_anaemic",
            "children_age_group_6_to_59_months_who_are_anaemic": "child_anaemic",
            "men_age_group_15_to_49_years_who_are_anaemic": "men_anaemic",
            
            "women_with_body_mass_index_bmi_below_normal": "women_bmi_below_normal",
            "men_with_body_mass_index_bmi_below_normal": "men_bmi_below_normal",
            "women_who_are_overweight_or_obese": "women_overweight_obese",
            "men_who_are_overweight_or_obese": "men_overweight_obese",
            
            "women_suffering_from_very_high_blood_sugar": "women_vhigh_blood_sugar",
            "women_suffering_from_high_blood_sugar": "women_high_blood_sugar",
            "men_suffering_from_very_high_blood_sugar": "men_vhigh_blood_sugar",
            "men_suffering_from_high_blood_sugar": "men_high_blood_sugar",
            
            "women_with_moderately_high_hypertension": "women_mod_high_bp",
            "women_moderately_or_severely_elevated_blood_pressure": "women_mod_high_bp",
            "women_mildly_elevated_blood_pressure": "women_mild_bp",
            "men_with_moderately_high_hypertension": "men_mod_high_bp",
            "men_moderately_or_severely_elevated_blood_pressure": "men_mod_high_bp",
            "men_mildly_elevated_blood_pressure": "men_mild_bp",
            
            "clean_fuel_for_cooking": "hh_clean_cooking_fuel",
            "improved_sanitation_facility": "hh_improved_sanitation",
            "improved_drinking_water": "hh_improved_drinking_water",
            "with_electricity": "hh_electricity",
            "iodized_salt": "hh_iodized_salt",
            "health_insurance": "hh_health_insurance",
            
            "10_or_more_years_of_schooling": "women_10yrs_schooling",
            "ever_attended_school": "female_pop_attended_school",
            "women_age_group_15_to_49_years_who_are_literate": "women_literate",
            "men_age_group_15_to_49_years_who_are_literate": "men_literate",
            
            "women_in_the_age_group_of_20_to_24_years_married_before_age_18": "women_married_before_18",
            "women_age_group_20_to_24_years_married_before_age_18": "women_married_before_18",
            "women_age_group_15_to_19_years_who_were_already_mothers_or_pregnant": "women_15_19_mothers",
            
            "any_modern_family_planning_method": "fp_modern_method",
            "use_any_family_planning_methods": "fp_any_method",
            "unmet_need_for_family_planning": "fp_unmet_need",
            "female_sterilization": "fp_female_steril",
            "male_sterlization": "fp_male_steril",
            "consuming_pill": "fp_pill",
            "using_condom": "fp_condom",
            "using_intrauterine_device": "fp_iud",
            
            "antenatal_check_up_in_the_first_trimester": "anc_first_trimester",
            "at_least_4_antenatal_care_visits": "anc_4_plus_visits",
            "institutional_births_in_public_facility": "institutional_births_public",
            "institutional_births": "institutional_births_total",
            "caesarean_section": "births_c_section",
            
            "fully_immunized": "child_fully_immunized",
            "fully_vaccinated": "child_fully_immunized",
            "received_bacillus_calmette_guerin_bcg": "child_bcg",
            "3_doses_of_polio": "child_polio_3",
            "3_doses_of_dpt": "child_dpt_3",
            "first_dose_of_measles": "child_measles",
            "3_doses_of_measles": "child_measles",
            "vitamin_a_dose": "child_vit_a",
            
            "prevalence_of_diarrhoea": "child_diarrhoea",
            "diarrhoea_in_the_2_weeks_preceding_the_survey_who_received_oral_rehydration_salts": "child_diarrhoea_ors",
            "diarrhoea_in_the_2_weeks_preceding_the_survey_who_received_zinc": "child_diarrhoea_zinc",
            "acute_respiratory_infection": "child_ari",
            
            "exclusively_breastfed": "child_excl_breastfed_6m",
            "breastfed_within_one_hour": "child_breastfed_1hr",
            
            "cervix_examination": "women_cervix_exam",
            "cervical_cancer": "women_cervix_exam",
            "breast_examination": "women_breast_exam",
            "oral_cavity": "women_oral_exam"
        }

    @staticmethod
    def _clean_column_name(col: str) -> str:
        """Removes NDAP noise and standardizes column string."""
        col = re.sub(r"\s*\(UOM:.*?\)\s*\|Scaling Factor:\d+", "", str(col))
        col = re.sub(r"\s*\(UOM:.*$", "", col)
        col = col.lower().strip()
        col = re.sub(r"[^\w\s-]", "", col)
        col = re.sub(r"[\s-]+", "_", col)
        return col

    @staticmethod
    def _extract_year(year_str: str) -> int | None:
        match = re.search(r"(\d{4})", str(year_str))
        return int(match.group(1)) if match else None

    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies exact mapping logic and deduplicates columns."""
        new_cols = {}
        for col in df.columns:
            clean_name = self._clean_column_name(col)
            matched = False
            for key, short_name in self.metric_mapping.items():
                if key in clean_name:
                    new_cols[col] = short_name
                    matched = True
                    break
            if not matched:
                new_cols[col] = clean_name
        
        df = df.rename(columns=new_cols)
        # CRITICAL FIX: Drop accidentally duplicated columns caused by renaming collisions
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        return df

    def process_nfhs(self):
        logger.info("Starting NFHS Data Processing...")
        try:
            df4 = pd.read_csv(self.raw_dir / "nfhs4_district/NDAP_REPORT_7034.csv")
            df5 = pd.read_csv(self.raw_dir / "nfhs5_district/NDAP_REPORT_6822.csv")
            
            if 'Residence type' in df4.columns:
                df4 = df4[df4['Residence type'] == 'Total'].drop(columns=['Residence type'])
                
            df4 = self._map_columns(df4)
            df5 = self._map_columns(df5)
            
            df4['year'] = df4['year'].apply(self._extract_year)
            df5['year'] = df5['year'].apply(self._extract_year)
            
            identifiers =['state_lgd_code', 'state', 'district_lgd_code', 'district', 'year']
            
            # Find the true intersection of our cleaned, mapped, and deduplicated columns
            metrics_4 = set(df4.columns) - set(identifiers)
            metrics_5 = set(df5.columns) - set(identifiers)
            intersecting_metrics = list(metrics_4.intersection(metrics_5))
            
            logger.info(f"Explicit mapping resulted in {len(intersecting_metrics)} perfectly matched metrics.")
            
            columns_to_keep = identifiers + intersecting_metrics
            df_pooled_wide = pd.concat([df4[columns_to_keep], df5[columns_to_keep]], ignore_index=True)
            df_pooled_wide[intersecting_metrics] = df_pooled_wide[intersecting_metrics].apply(pd.to_numeric, errors='coerce')
            
            df_pooled_long = df_pooled_wide.melt(
                id_vars=identifiers, 
                value_vars=intersecting_metrics,
                var_name='metric_name', 
                value_name='metric_value'
            ).dropna(subset=['metric_value'])
            
            df_pooled_wide.to_parquet(self.tabular_out / "nfhs_pooled_wide.parquet", index=False)
            df_pooled_long.to_parquet(self.tabular_out / "nfhs_pooled_long.parquet", index=False)
            logger.info("Successfully pooled and saved NFHS datasets.")
            
        except Exception as e:
            logger.error(f"Error processing NFHS: {str(e)}", exc_info=True)
            raise

    def process_spatial_concordance(self):
        logger.info("Starting Spatial Concordance Mapping...")
        try:
            gdf = gpd.read_file(self.raw_dir / "india_districts.geojson")
            concordance = pd.read_csv(self.raw_dir / "district_concordance_with_LGD_codes_parent.csv")
            
            gdf['dt_code_int'] = pd.to_numeric(gdf['dt_code'], errors='coerce')
            concordance['Census 2011 Code Int'] = pd.to_numeric(concordance['Census 2011 Code'], errors='coerce')
            
            gdf = gdf.dropna(subset=['dt_code_int'])
            concordance = concordance.dropna(subset=['Census 2011 Code Int'])
            
            merged_gdf = gdf.merge(
                concordance[['Census 2011 Code Int', 'LGD District Code', 'LGD State Code', 'LGD District Name']], 
                left_on='dt_code_int', right_on='Census 2011 Code Int', how='inner'
            )
            
            columns_to_keep =['LGD District Code', 'LGD State Code', 'LGD District Name', 'st_nm', 'geometry']
            merged_gdf = merged_gdf[columns_to_keep]
            merged_gdf.columns =['district_lgd_code', 'state_lgd_code', 'district_name', 'state_name', 'geometry']
            
            merged_gdf.to_file(self.spatial_out / "india_districts_lgd.geojson", driver='GeoJSON')
            logger.info("Successfully mapped and saved Spatial bounds.")
        except Exception as e:
            logger.error(f"Error in Spatial Concordance: {str(e)}", exc_info=True)
            raise

    def process_infrastructure(self):
        logger.info("Starting Infrastructure Data Processing...")
        try:
            df_pmgsy = pd.read_csv(self.raw_dir / "pmgsy/NDAP_REPORT_7096.csv")
            df_pmgsy.columns =[self._clean_column_name(c) for c in df_pmgsy.columns]
            df_pmgsy['year'] = df_pmgsy['yearcode'].astype(int)
            
            pmgsy_agg = df_pmgsy.groupby(['district_lgd_code', 'year']).agg({
                'road_length_of_state_and_district': 'sum',
                'sanction_cost_granted_by_ministry_of_rural_development_to_for_road_construction': 'sum'
            }).reset_index()
            pmgsy_agg.rename(columns={
                'road_length_of_state_and_district': 'pmgsy_road_length_km',
                'sanction_cost_granted_by_ministry_of_rural_development_to_for_road_construction': 'pmgsy_sanction_cost'
            }, inplace=True)
            pmgsy_agg.to_parquet(self.tabular_out / "pmgsy_district_agg.parquet", index=False)
            
            df_mgnrega = pd.read_csv(self.raw_dir / "mgnrega/NDAP_REPORT_6026.csv")
            df_mgnrega.columns =[self._clean_column_name(c) for c in df_mgnrega.columns]
            df_mgnrega['year'] = df_mgnrega['yearcode'].astype(int)
            
            cols_to_keep =['state_lgd_code', 'state', 'year', 'households_that_demanded_work', 'labour_expenditure_that_has_been_disbursed']
            mgnrega_clean = df_mgnrega[cols_to_keep].rename(columns={
                'households_that_demanded_work': 'mgnrega_demand_households',
                'labour_expenditure_that_has_been_disbursed': 'mgnrega_labour_exp'
            })
            mgnrega_clean.to_parquet(self.tabular_out / "mgnrega_state_clean.parquet", index=False)
            logger.info("Successfully processed PMGSY and MGNREGA.")
        except Exception as e:
            logger.error(f"Error processing Infrastructure: {str(e)}", exc_info=True)
            raise

    def run_all(self):
        logger.info("--- Initiating Full Tabular Ingestion Pipeline ---")
        self.process_nfhs()
        self.process_spatial_concordance()
        self.process_infrastructure()
        logger.info("--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    pipeline.run_all()