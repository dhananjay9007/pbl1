import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
import altair as alt

@st.cache_data
def load_data(path='survey_data.csv.csv'):
    return pd.read_csv(path)

st.set_page_config(layout="wide", page_title="Survey Analytics Dashboard")

st.title("Survey Analytics — Classification, Association Rules, Clustering, Regression")

# Load data
df = load_data('/mnt/data/streamlit_data/survey_data.csv.csv') if False else load_data('/mnt/data/survey_data.csv.csv')

with st.sidebar:
    st.header("Filters")
    _min, _max = float(df['response_id'].min()), float(df['response_id'].max())
    response_id_range = st.slider('response_id range', _min, _max, (_min, _max))
    age_group_choices = st.multiselect('age_group', options=df['age_group'].dropna().unique().tolist(), default=df['age_group'].dropna().unique().tolist())
    gender_choices = st.multiselect('gender', options=df['gender'].dropna().unique().tolist(), default=df['gender'].dropna().unique().tolist())
    employment_status_choices = st.multiselect('employment_status', options=df['employment_status'].dropna().unique().tolist(), default=df['employment_status'].dropna().unique().tolist())
    income_choices = st.multiselect('income', options=df['income'].dropna().unique().tolist(), default=df['income'].dropna().unique().tolist())
    education_choices = st.multiselect('education', options=df['education'].dropna().unique().tolist(), default=df['education'].dropna().unique().tolist())
    location_type_choices = st.multiselect('location_type', options=df['location_type'].dropna().unique().tolist(), default=df['location_type'].dropna().unique().tolist())
    household_size_choices = st.multiselect('household_size', options=df['household_size'].dropna().unique().tolist(), default=df['household_size'].dropna().unique().tolist())
    _min, _max = float(df['health_consciousness'].min()), float(df['health_consciousness'].max())
    health_consciousness_range = st.slider('health_consciousness range', _min, _max, (_min, _max))
    exercise_frequency_choices = st.multiselect('exercise_frequency', options=df['exercise_frequency'].dropna().unique().tolist(), default=df['exercise_frequency'].dropna().unique().tolist())
    fitness_goal_choices = st.multiselect('fitness_goal', options=df['fitness_goal'].dropna().unique().tolist(), default=df['fitness_goal'].dropna().unique().tolist())
    _min, _max = float(df['hydration_importance'].min()), float(df['hydration_importance'].max())
    hydration_importance_range = st.slider('hydration_importance range', _min, _max, (_min, _max))
    daily_water_intake_choices = st.multiselect('daily_water_intake', options=df['daily_water_intake'].dropna().unique().tolist(), default=df['daily_water_intake'].dropna().unique().tolist())
    bottle_type_choices = st.multiselect('bottle_type', options=df['bottle_type'].dropna().unique().tolist(), default=df['bottle_type'].dropna().unique().tolist())
    monthly_beverage_spend_choices = st.multiselect('monthly_beverage_spend', options=df['monthly_beverage_spend'].dropna().unique().tolist(), default=df['monthly_beverage_spend'].dropna().unique().tolist())
    _min, _max = float(df['interest_level'].min()), float(df['interest_level'].max())
    interest_level_range = st.slider('interest_level range', _min, _max, (_min, _max))
    purchase_likelihood_choices = st.multiselect('purchase_likelihood', options=df['purchase_likelihood'].dropna().unique().tolist(), default=df['purchase_likelihood'].dropna().unique().tolist())
    _min, _max = float(df['willingness_to_pay_continuous'].min()), float(df['willingness_to_pay_continuous'].max())
    willingness_to_pay_continuous_range = st.slider('willingness_to_pay_continuous range', _min, _max, (_min, _max))
    willingness_to_pay_category_choices = st.multiselect('willingness_to_pay_category', options=df['willingness_to_pay_category'].dropna().unique().tolist(), default=df['willingness_to_pay_category'].dropna().unique().tolist())
    weekly_usage_choices = st.multiselect('weekly_usage', options=df['weekly_usage'].dropna().unique().tolist(), default=df['weekly_usage'].dropna().unique().tolist())
    purchase_preference_choices = st.multiselect('purchase_preference', options=df['purchase_preference'].dropna().unique().tolist(), default=df['purchase_preference'].dropna().unique().tolist())
    _min, _max = float(df['sustainability_importance'].min()), float(df['sustainability_importance'].max())
    sustainability_importance_range = st.slider('sustainability_importance range', _min, _max, (_min, _max))
    _min, _max = float(df['early_adopter_score'].min()), float(df['early_adopter_score'].max())
    early_adopter_score_range = st.slider('early_adopter_score range', _min, _max, (_min, _max))
    _min, _max = float(df['premium_willingness_score'].min()), float(df['premium_willingness_score'].max())
    premium_willingness_score_range = st.slider('premium_willingness_score range', _min, _max, (_min, _max))
    _min, _max = float(df['health_condition_Diabetes (Type 1 or 2)'].min()), float(df['health_condition_Diabetes (Type 1 or 2)'].max())
    health_condition_Diabetes (Type 1 or 2)_range = st.slider('health_condition_Diabetes (Type 1 or 2) range', _min, _max, (_min, _max))
    _min, _max = float(df['health_condition_High blood pressure'].min()), float(df['health_condition_High blood pressure'].max())
    health_condition_High blood pressure_range = st.slider('health_condition_High blood pressure range', _min, _max, (_min, _max))
    _min, _max = float(df['health_condition_High cholesterol'].min()), float(df['health_condition_High cholesterol'].max())
    health_condition_High cholesterol_range = st.slider('health_condition_High cholesterol range', _min, _max, (_min, _max))
    _min, _max = float(df['health_condition_Heart disease'].min()), float(df['health_condition_Heart disease'].max())
    health_condition_Heart disease_range = st.slider('health_condition_Heart disease range', _min, _max, (_min, _max))
    _min, _max = float(df['health_condition_Kidney disease'].min()), float(df['health_condition_Kidney disease'].max())
    health_condition_Kidney disease_range = st.slider('health_condition_Kidney disease range', _min, _max, (_min, _max))
    _min, _max = float(df['health_condition_Food allergies'].min()), float(df['health_condition_Food allergies'].max())
    health_condition_Food allergies_range = st.slider('health_condition_Food allergies range', _min, _max, (_min, _max))
    _min, _max = float(df['health_condition_None of the above'].min()), float(df['health_condition_None of the above'].max())
    health_condition_None of the above_range = st.slider('health_condition_None of the above range', _min, _max, (_min, _max))
    _min, _max = float(df['barrier_I forget to drink'].min()), float(df['barrier_I forget to drink'].max())
    barrier_I forget to drink_range = st.slider('barrier_I forget to drink range', _min, _max, (_min, _max))
    _min, _max = float(df['barrier_Plain water is boring/tasteless'].min()), float(df['barrier_Plain water is boring/tasteless'].max())
    barrier_Plain water is boring/tasteless_range = st.slider('barrier_Plain water is boring/tasteless range', _min, _max, (_min, _max))
    _min, _max = float(df['barrier_I dont feel thirsty'].min()), float(df['barrier_I dont feel thirsty'].max())
    barrier_I dont feel thirsty_range = st.slider('barrier_I dont feel thirsty range', _min, _max, (_min, _max))
    _min, _max = float(df['barrier_Its inconvenient to carry water'].min()), float(df['barrier_Its inconvenient to carry water'].max())
    barrier_Its inconvenient to carry water_range = st.slider('barrier_Its inconvenient to carry water range', _min, _max, (_min, _max))
    _min, _max = float(df['barrier_I prefer other beverages'].min()), float(df['barrier_I prefer other beverages'].max())
    barrier_I prefer other beverages_range = st.slider('barrier_I prefer other beverages range', _min, _max, (_min, _max))
    _min, _max = float(df['barrier_Health reasons (frequent bathroom trips, etc.)'].min()), float(df['barrier_Health reasons (frequent bathroom trips, etc.)'].max())
    barrier_Health reasons (frequent bathroom trips, etc.)_range = st.slider('barrier_Health reasons (frequent bathroom trips, etc.) range', _min, _max, (_min, _max))
    _min, _max = float(df['barrier_Nothing prevents me'].min()), float(df['barrier_Nothing prevents me'].max())
    barrier_Nothing prevents me_range = st.slider('barrier_Nothing prevents me range', _min, _max, (_min, _max))
    _min, _max = float(df['consume_location_At home'].min()), float(df['consume_location_At home'].max())
    consume_location_At home_range = st.slider('consume_location_At home range', _min, _max, (_min, _max))
    _min, _max = float(df['consume_location_At work/school'].min()), float(df['consume_location_At work/school'].max())
    consume_location_At work/school_range = st.slider('consume_location_At work/school range', _min, _max, (_min, _max))
    _min, _max = float(df['consume_location_At the gym'].min()), float(df['consume_location_At the gym'].max())
    consume_location_At the gym_range = st.slider('consume_location_At the gym range', _min, _max, (_min, _max))
    _min, _max = float(df['consume_location_During commute'].min()), float(df['consume_location_During commute'].max())
    consume_location_During commute_range = st.slider('consume_location_During commute range', _min, _max, (_min, _max))
    _min, _max = float(df['consume_location_Restaurants/cafes'].min()), float(df['consume_location_Restaurants/cafes'].max())
    consume_location_Restaurants/cafes_range = st.slider('consume_location_Restaurants/cafes range', _min, _max, (_min, _max))
    _min, _max = float(df['consume_location_Outdoor activities'].min()), float(df['consume_location_Outdoor activities'].max())
    consume_location_Outdoor activities_range = st.slider('consume_location_Outdoor activities range', _min, _max, (_min, _max))
    _min, _max = float(df['beverage_Plain water'].min()), float(df['beverage_Plain water'].max())
    beverage_Plain water_range = st.slider('beverage_Plain water range', _min, _max, (_min, _max))
    _min, _max = float(df['beverage_Flavored water (e.g., LaCroix, Hint)'].min()), float(df['beverage_Flavored water (e.g., LaCroix, Hint)'].max())
    beverage_Flavored water (e.g., LaCroix, Hint)_range = st.slider('beverage_Flavored water (e.g., LaCroix, Hint) range', _min, _max, (_min, _max))
    _min, _max = float(df['beverage_Sports drinks (e.g., Gatorade, Powerade)'].min()), float(df['beverage_Sports drinks (e.g., Gatorade, Powerade)'].max())
    beverage_Sports drinks (e.g., Gatorade, Powerade)_range = st.slider('beverage_Sports drinks (e.g., Gatorade, Powerade) range', _min, _max, (_min, _max))
    _min, _max = float(df['beverage_Energy drinks (e.g., Red Bull, Monster)'].min()), float(df['beverage_Energy drinks (e.g., Red Bull, Monster)'].max())
    beverage_Energy drinks (e.g., Red Bull, Monster)_range = st.slider('beverage_Energy drinks (e.g., Red Bull, Monster) range', _min, _max, (_min, _max))
    _min, _max = float(df['beverage_Enhanced water (e.g., Vitaminwater, Smartwater)'].min()), float(df['beverage_Enhanced water (e.g., Vitaminwater, Smartwater)'].max())
    beverage_Enhanced water (e.g., Vitaminwater, Smartwater)_range = st.slider('beverage_Enhanced water (e.g., Vitaminwater, Smartwater) range', _min, _max, (_min, _max))
    _min, _max = float(df['beverage_Coffee'].min()), float(df['beverage_Coffee'].max())
    beverage_Coffee_range = st.slider('beverage_Coffee range', _min, _max, (_min, _max))
    _min, _max = float(df['beverage_Tea'].min()), float(df['beverage_Tea'].max())
    beverage_Tea_range = st.slider('beverage_Tea range', _min, _max, (_min, _max))
    _min, _max = float(df['beverage_Soda/soft drinks'].min()), float(df['beverage_Soda/soft drinks'].max())
    beverage_Soda/soft drinks_range = st.slider('beverage_Soda/soft drinks range', _min, _max, (_min, _max))
    _min, _max = float(df['beverage_Juice'].min()), float(df['beverage_Juice'].max())
    beverage_Juice_range = st.slider('beverage_Juice range', _min, _max, (_min, _max))
    _min, _max = float(df['beverage_Protein shakes'].min()), float(df['beverage_Protein shakes'].max())
    beverage_Protein shakes_range = st.slider('beverage_Protein shakes range', _min, _max, (_min, _max))
    _min, _max = float(df['beverage_Pre-workout drinks'].min()), float(df['beverage_Pre-workout drinks'].max())
    beverage_Pre-workout drinks_range = st.slider('beverage_Pre-workout drinks range', _min, _max, (_min, _max))
    _min, _max = float(df['beverage_Electrolyte tablets/powders (e.g., Nuun, Liquid I.V.)'].min()), float(df['beverage_Electrolyte tablets/powders (e.g., Nuun, Liquid I.V.)'].max())
    beverage_Electrolyte tablets/powders (e.g., Nuun, Liquid I.V.)_range = st.slider('beverage_Electrolyte tablets/powders (e.g., Nuun, Liquid I.V.) range', _min, _max, (_min, _max))
    _min, _max = float(df['beverage_Squeeze flavor enhancers (e.g., MiO, Crystal Light)'].min()), float(df['beverage_Squeeze flavor enhancers (e.g., MiO, Crystal Light)'].max())
    beverage_Squeeze flavor enhancers (e.g., MiO, Crystal Light)_range = st.slider('beverage_Squeeze flavor enhancers (e.g., MiO, Crystal Light) range', _min, _max, (_min, _max))
    _min, _max = float(df['beverage_Kombucha'].min()), float(df['beverage_Kombucha'].max())
    beverage_Kombucha_range = st.slider('beverage_Kombucha range', _min, _max, (_min, _max))
    _min, _max = float(df['beverage_Coconut water'].min()), float(df['beverage_Coconut water'].max())
    beverage_Coconut water_range = st.slider('beverage_Coconut water range', _min, _max, (_min, _max))
    _min, _max = float(df['priority_Taste/flavor variety'].min()), float(df['priority_Taste/flavor variety'].max())
    priority_Taste/flavor variety_range = st.slider('priority_Taste/flavor variety range', _min, _max, (_min, _max))
    _min, _max = float(df['priority_Nutritional benefits (vitamins, minerals)'].min()), float(df['priority_Nutritional benefits (vitamins, minerals)'].max())
    priority_Nutritional benefits (vitamins, minerals)_range = st.slider('priority_Nutritional benefits (vitamins, minerals) range', _min, _max, (_min, _max))
    _min, _max = float(df['priority_Low/zero sugar'].min()), float(df['priority_Low/zero sugar'].max())
    priority_Low/zero sugar_range = st.slider('priority_Low/zero sugar range', _min, _max, (_min, _max))
    _min, _max = float(df['priority_Natural ingredients'].min()), float(df['priority_Natural ingredients'].max())
    priority_Natural ingredients_range = st.slider('priority_Natural ingredients range', _min, _max, (_min, _max))
    _min, _max = float(df['priority_Caffeine content'].min()), float(df['priority_Caffeine content'].max())
    priority_Caffeine content_range = st.slider('priority_Caffeine content range', _min, _max, (_min, _max))
    _min, _max = float(df['priority_Electrolytes'].min()), float(df['priority_Electrolytes'].max())
    priority_Electrolytes_range = st.slider('priority_Electrolytes range', _min, _max, (_min, _max))
    _min, _max = float(df['priority_Protein content'].min()), float(df['priority_Protein content'].max())
    priority_Protein content_range = st.slider('priority_Protein content range', _min, _max, (_min, _max))
    _min, _max = float(df['priority_Convenience/portability'].min()), float(df['priority_Convenience/portability'].max())
    priority_Convenience/portability_range = st.slider('priority_Convenience/portability range', _min, _max, (_min, _max))
    _min, _max = float(df['priority_Price/affordability'].min()), float(df['priority_Price/affordability'].max())
    priority_Price/affordability_range = st.slider('priority_Price/affordability range', _min, _max, (_min, _max))
    _min, _max = float(df['priority_Brand reputation'].min()), float(df['priority_Brand reputation'].max())
    priority_Brand reputation_range = st.slider('priority_Brand reputation range', _min, _max, (_min, _max))
    _min, _max = float(df['priority_Environmental sustainability'].min()), float(df['priority_Environmental sustainability'].max())
    priority_Environmental sustainability_range = st.slider('priority_Environmental sustainability range', _min, _max, (_min, _max))
    _min, _max = float(df['priority_No artificial ingredients'].min()), float(df['priority_No artificial ingredients'].max())
    priority_No artificial ingredients_range = st.slider('priority_No artificial ingredients range', _min, _max, (_min, _max))
    _min, _max = float(df['used_product_Cirkul (flavor cartridge bottle)'].min()), float(df['used_product_Cirkul (flavor cartridge bottle)'].max())
    used_product_Cirkul (flavor cartridge bottle)_range = st.slider('used_product_Cirkul (flavor cartridge bottle) range', _min, _max, (_min, _max))
    _min, _max = float(df['used_product_Air Up (scent-based bottle)'].min()), float(df['used_product_Air Up (scent-based bottle)'].max())
    used_product_Air Up (scent-based bottle)_range = st.slider('used_product_Air Up (scent-based bottle) range', _min, _max, (_min, _max))
    _min, _max = float(df['used_product_MiO/liquid flavor drops'].min()), float(df['used_product_MiO/liquid flavor drops'].max())
    used_product_MiO/liquid flavor drops_range = st.slider('used_product_MiO/liquid flavor drops range', _min, _max, (_min, _max))
    _min, _max = float(df['used_product_Dissolvable tablets (Nuun, Liquid I.V.)'].min()), float(df['used_product_Dissolvable tablets (Nuun, Liquid I.V.)'].max())
    used_product_Dissolvable tablets (Nuun, Liquid I.V.)_range = st.slider('used_product_Dissolvable tablets (Nuun, Liquid I.V.) range', _min, _max, (_min, _max))
    _min, _max = float(df['used_product_Powder packets (Crystal Light, etc.)'].min()), float(df['used_product_Powder packets (Crystal Light, etc.)'].max())
    used_product_Powder packets (Crystal Light, etc.)_range = st.slider('used_product_Powder packets (Crystal Light, etc.) range', _min, _max, (_min, _max))
    _min, _max = float(df['used_product_Pre-flavored bottled water'].min()), float(df['used_product_Pre-flavored bottled water'].max())
    used_product_Pre-flavored bottled water_range = st.slider('used_product_Pre-flavored bottled water range', _min, _max, (_min, _max))
    _min, _max = float(df['used_product_None of the above'].min()), float(df['used_product_None of the above'].max())
    used_product_None of the above_range = st.slider('used_product_None of the above range', _min, _max, (_min, _max))
    _min, _max = float(df['appealing_Convenience/ease of use'].min()), float(df['appealing_Convenience/ease of use'].max())
    appealing_Convenience/ease of use_range = st.slider('appealing_Convenience/ease of use range', _min, _max, (_min, _max))
    _min, _max = float(df['appealing_Portion control'].min()), float(df['appealing_Portion control'].max())
    appealing_Portion control_range = st.slider('appealing_Portion control range', _min, _max, (_min, _max))
    _min, _max = float(df['appealing_No mixing required'].min()), float(df['appealing_No mixing required'].max())
    appealing_No mixing required_range = st.slider('appealing_No mixing required range', _min, _max, (_min, _max))
    _min, _max = float(df['appealing_Flavor variety'].min()), float(df['appealing_Flavor variety'].max())
    appealing_Flavor variety_range = st.slider('appealing_Flavor variety range', _min, _max, (_min, _max))
    _min, _max = float(df['appealing_Nutritional customization'].min()), float(df['appealing_Nutritional customization'].max())
    appealing_Nutritional customization_range = st.slider('appealing_Nutritional customization range', _min, _max, (_min, _max))
    _min, _max = float(df['appealing_Portability'].min()), float(df['appealing_Portability'].max())
    appealing_Portability_range = st.slider('appealing_Portability range', _min, _max, (_min, _max))
    _min, _max = float(df['appealing_Hygiene (single-use)'].min()), float(df['appealing_Hygiene (single-use)'].max())
    appealing_Hygiene (single-use)_range = st.slider('appealing_Hygiene (single-use) range', _min, _max, (_min, _max))
    _min, _max = float(df['appealing_Helps me drink more water'].min()), float(df['appealing_Helps me drink more water'].max())
    appealing_Helps me drink more water_range = st.slider('appealing_Helps me drink more water range', _min, _max, (_min, _max))
    _min, _max = float(df['appealing_Health benefits'].min()), float(df['appealing_Health benefits'].max())
    appealing_Health benefits_range = st.slider('appealing_Health benefits range', _min, _max, (_min, _max))
    _min, _max = float(df['appealing_Nothing appeals to me'].min()), float(df['appealing_Nothing appeals to me'].max())
    appealing_Nothing appeals to me_range = st.slider('appealing_Nothing appeals to me range', _min, _max, (_min, _max))
    _min, _max = float(df['concern_Price/cost per use'].min()), float(df['concern_Price/cost per use'].max())
    concern_Price/cost per use_range = st.slider('concern_Price/cost per use range', _min, _max, (_min, _max))
    _min, _max = float(df['concern_Environmental waste (single-use plastic)'].min()), float(df['concern_Environmental waste (single-use plastic)'].max())
    concern_Environmental waste (single-use plastic)_range = st.slider('concern_Environmental waste (single-use plastic) range', _min, _max, (_min, _max))
    _min, _max = float(df['concern_Powder might not dissolve well'].min()), float(df['concern_Powder might not dissolve well'].max())
    concern_Powder might not dissolve well_range = st.slider('concern_Powder might not dissolve well range', _min, _max, (_min, _max))
    _min, _max = float(df['concern_Artificial ingredients'].min()), float(df['concern_Artificial ingredients'].max())
    concern_Artificial ingredients_range = st.slider('concern_Artificial ingredients range', _min, _max, (_min, _max))
    _min, _max = float(df['concern_Limited flavor options'].min()), float(df['concern_Limited flavor options'].max())
    concern_Limited flavor options_range = st.slider('concern_Limited flavor options range', _min, _max, (_min, _max))
    _min, _max = float(df['concern_Not compatible with my bottle'].min()), float(df['concern_Not compatible with my bottle'].max())
    concern_Not compatible with my bottle_range = st.slider('concern_Not compatible with my bottle range', _min, _max, (_min, _max))
    _min, _max = float(df['concern_Prefer other methods'].min()), float(df['concern_Prefer other methods'].max())
    concern_Prefer other methods_range = st.slider('concern_Prefer other methods range', _min, _max, (_min, _max))
    _min, _max = float(df['concern_No concerns'].min()), float(df['concern_No concerns'].max())
    concern_No concerns_range = st.slider('concern_No concerns range', _min, _max, (_min, _max))
    price_15_perception_choices = st.multiselect('price_15_perception', options=df['price_15_perception'].dropna().unique().tolist(), default=df['price_15_perception'].dropna().unique().tolist())
    price_25_perception_choices = st.multiselect('price_25_perception', options=df['price_25_perception'].dropna().unique().tolist(), default=df['price_25_perception'].dropna().unique().tolist())
    price_35_perception_choices = st.multiselect('price_35_perception', options=df['price_35_perception'].dropna().unique().tolist(), default=df['price_35_perception'].dropna().unique().tolist())
    price_45_perception_choices = st.multiselect('price_45_perception', options=df['price_45_perception'].dropna().unique().tolist(), default=df['price_45_perception'].dropna().unique().tolist())
    price_60_perception_choices = st.multiselect('price_60_perception', options=df['price_60_perception'].dropna().unique().tolist(), default=df['price_60_perception'].dropna().unique().tolist())
    _min, _max = float(df['flavor_Citrus (lemon, lime, orange)'].min()), float(df['flavor_Citrus (lemon, lime, orange)'].max())
    flavor_Citrus (lemon, lime, orange)_range = st.slider('flavor_Citrus (lemon, lime, orange) range', _min, _max, (_min, _max))
    _min, _max = float(df['flavor_Berry (strawberry, blueberry, raspberry)'].min()), float(df['flavor_Berry (strawberry, blueberry, raspberry)'].max())
    flavor_Berry (strawberry, blueberry, raspberry)_range = st.slider('flavor_Berry (strawberry, blueberry, raspberry) range', _min, _max, (_min, _max))
    _min, _max = float(df['flavor_Tropical (mango, pineapple, coconut)'].min()), float(df['flavor_Tropical (mango, pineapple, coconut)'].max())
    flavor_Tropical (mango, pineapple, coconut)_range = st.slider('flavor_Tropical (mango, pineapple, coconut) range', _min, _max, (_min, _max))
    _min, _max = float(df['flavor_Mint/herbal'].min()), float(df['flavor_Mint/herbal'].max())
    flavor_Mint/herbal_range = st.slider('flavor_Mint/herbal range', _min, _max, (_min, _max))
    _min, _max = float(df['flavor_Green tea/matcha'].min()), float(df['flavor_Green tea/matcha'].max())
    flavor_Green tea/matcha_range = st.slider('flavor_Green tea/matcha range', _min, _max, (_min, _max))
    _min, _max = float(df['flavor_Coffee-flavored'].min()), float(df['flavor_Coffee-flavored'].max())
    flavor_Coffee-flavored_range = st.slider('flavor_Coffee-flavored range', _min, _max, (_min, _max))
    _min, _max = float(df['flavor_Neutral/unflavored (just nutrients)'].min()), float(df['flavor_Neutral/unflavored (just nutrients)'].max())
    flavor_Neutral/unflavored (just nutrients)_range = st.slider('flavor_Neutral/unflavored (just nutrients) range', _min, _max, (_min, _max))
    _min, _max = float(df['flavor_Sour/tangy'].min()), float(df['flavor_Sour/tangy'].max())
    flavor_Sour/tangy_range = st.slider('flavor_Sour/tangy range', _min, _max, (_min, _max))
    _min, _max = float(df['flavor_Sweet/dessert-inspired'].min()), float(df['flavor_Sweet/dessert-inspired'].max())
    flavor_Sweet/dessert-inspired_range = st.slider('flavor_Sweet/dessert-inspired range', _min, _max, (_min, _max))
    _min, _max = float(df['benefit_rank_Energy boost (caffeine, B-vitamins)'].min()), float(df['benefit_rank_Energy boost (caffeine, B-vitamins)'].max())
    benefit_rank_Energy boost (caffeine, B-vitamins)_range = st.slider('benefit_rank_Energy boost (caffeine, B-vitamins) range', _min, _max, (_min, _max))
    _min, _max = float(df['benefit_rank_Hydration/electrolytes'].min()), float(df['benefit_rank_Hydration/electrolytes'].max())
    benefit_rank_Hydration/electrolytes_range = st.slider('benefit_rank_Hydration/electrolytes range', _min, _max, (_min, _max))
    _min, _max = float(df['benefit_rank_Immunity support (Vitamin C, zinc)'].min()), float(df['benefit_rank_Immunity support (Vitamin C, zinc)'].max())
    benefit_rank_Immunity support (Vitamin C, zinc)_range = st.slider('benefit_rank_Immunity support (Vitamin C, zinc) range', _min, _max, (_min, _max))
    _min, _max = float(df['benefit_rank_Focus/mental clarity'].min()), float(df['benefit_rank_Focus/mental clarity'].max())
    benefit_rank_Focus/mental clarity_range = st.slider('benefit_rank_Focus/mental clarity range', _min, _max, (_min, _max))
    _min, _max = float(df['benefit_rank_Recovery (post-workout)'].min()), float(df['benefit_rank_Recovery (post-workout)'].max())
    benefit_rank_Recovery (post-workout)_range = st.slider('benefit_rank_Recovery (post-workout) range', _min, _max, (_min, _max))
    _min, _max = float(df['benefit_rank_Digestive health (probiotics)'].min()), float(df['benefit_rank_Digestive health (probiotics)'].max())
    benefit_rank_Digestive health (probiotics)_range = st.slider('benefit_rank_Digestive health (probiotics) range', _min, _max, (_min, _max))
    _min, _max = float(df['benefit_rank_Antioxidants'].min()), float(df['benefit_rank_Antioxidants'].max())
    benefit_rank_Antioxidants_range = st.slider('benefit_rank_Antioxidants range', _min, _max, (_min, _max))
    _min, _max = float(df['benefit_rank_Skin health (collagen, biotin)'].min()), float(df['benefit_rank_Skin health (collagen, biotin)'].max())
    benefit_rank_Skin health (collagen, biotin)_range = st.slider('benefit_rank_Skin health (collagen, biotin) range', _min, _max, (_min, _max))
    _min, _max = float(df['benefit_rank_Weight management'].min()), float(df['benefit_rank_Weight management'].max())
    benefit_rank_Weight management_range = st.slider('benefit_rank_Weight management range', _min, _max, (_min, _max))
    _min, _max = float(df['benefit_rank_General vitamins/minerals'].min()), float(df['benefit_rank_General vitamins/minerals'].max())
    benefit_rank_General vitamins/minerals_range = st.slider('benefit_rank_General vitamins/minerals range', _min, _max, (_min, _max))
    _min, _max = float(df['specialized_Sugar-free (for diabetics)'].min()), float(df['specialized_Sugar-free (for diabetics)'].max())
    specialized_Sugar-free (for diabetics)_range = st.slider('specialized_Sugar-free (for diabetics) range', _min, _max, (_min, _max))
    _min, _max = float(df['specialized_Keto-friendly'].min()), float(df['specialized_Keto-friendly'].max())
    specialized_Keto-friendly_range = st.slider('specialized_Keto-friendly range', _min, _max, (_min, _max))
    _min, _max = float(df['specialized_Vegan'].min()), float(df['specialized_Vegan'].max())
    specialized_Vegan_range = st.slider('specialized_Vegan range', _min, _max, (_min, _max))
    _min, _max = float(df['specialized_Organic/natural only'].min()), float(df['specialized_Organic/natural only'].max())
    specialized_Organic/natural only_range = st.slider('specialized_Organic/natural only range', _min, _max, (_min, _max))
    _min, _max = float(df['specialized_Allergen-free (gluten, dairy, soy)'].min()), float(df['specialized_Allergen-free (gluten, dairy, soy)'].max())
    specialized_Allergen-free (gluten, dairy, soy)_range = st.slider('specialized_Allergen-free (gluten, dairy, soy) range', _min, _max, (_min, _max))
    _min, _max = float(df['specialized_Kid-friendly formulas'].min()), float(df['specialized_Kid-friendly formulas'].max())
    specialized_Kid-friendly formulas_range = st.slider('specialized_Kid-friendly formulas range', _min, _max, (_min, _max))
    _min, _max = float(df['specialized_Senior-optimized (bone health, etc.)'].min()), float(df['specialized_Senior-optimized (bone health, etc.)'].max())
    specialized_Senior-optimized (bone health, etc.)_range = st.slider('specialized_Senior-optimized (bone health, etc.) range', _min, _max, (_min, _max))
    _min, _max = float(df['specialized_Athletic performance'].min()), float(df['specialized_Athletic performance'].max())
    specialized_Athletic performance_range = st.slider('specialized_Athletic performance range', _min, _max, (_min, _max))
    _min, _max = float(df['specialized_None - prefer standard versions'].min()), float(df['specialized_None - prefer standard versions'].max())
    specialized_None - prefer standard versions_range = st.slider('specialized_None - prefer standard versions range', _min, _max, (_min, _max))
    _min, _max = float(df['discovery_Social media (Instagram, TikTok, Facebook)'].min()), float(df['discovery_Social media (Instagram, TikTok, Facebook)'].max())
    discovery_Social media (Instagram, TikTok, Facebook)_range = st.slider('discovery_Social media (Instagram, TikTok, Facebook) range', _min, _max, (_min, _max))
    _min, _max = float(df['discovery_Friends/family recommendations'].min()), float(df['discovery_Friends/family recommendations'].max())
    discovery_Friends/family recommendations_range = st.slider('discovery_Friends/family recommendations range', _min, _max, (_min, _max))
    _min, _max = float(df['discovery_Health/fitness influencers'].min()), float(df['discovery_Health/fitness influencers'].max())
    discovery_Health/fitness influencers_range = st.slider('discovery_Health/fitness influencers range', _min, _max, (_min, _max))
    _min, _max = float(df['discovery_Online reviews'].min()), float(df['discovery_Online reviews'].max())
    discovery_Online reviews_range = st.slider('discovery_Online reviews range', _min, _max, (_min, _max))
    _min, _max = float(df['discovery_In-store displays'].min()), float(df['discovery_In-store displays'].max())
    discovery_In-store displays_range = st.slider('discovery_In-store displays range', _min, _max, (_min, _max))
    _min, _max = float(df['discovery_TV/online ads'].min()), float(df['discovery_TV/online ads'].max())
    discovery_TV/online ads_range = st.slider('discovery_TV/online ads range', _min, _max, (_min, _max))
    _min, _max = float(df['discovery_Health blogs/websites'].min()), float(df['discovery_Health blogs/websites'].max())
    discovery_Health blogs/websites_range = st.slider('discovery_Health blogs/websites range', _min, _max, (_min, _max))
    _min, _max = float(df['discovery_Nutritionist/doctor recommendation'].min()), float(df['discovery_Nutritionist/doctor recommendation'].max())
    discovery_Nutritionist/doctor recommendation_range = st.slider('discovery_Nutritionist/doctor recommendation range', _min, _max, (_min, _max))

# Apply filters
df_filtered = df.copy()
df_filtered = df_filtered[(df_filtered['response_id']>= response_id_range[0]) & (df_filtered['response_id']<= response_id_range[1])]
if len(age_group_choices) > 0:
    df_filtered = df_filtered[df_filtered['age_group'].isin(age_group_choices)]
if len(gender_choices) > 0:
    df_filtered = df_filtered[df_filtered['gender'].isin(gender_choices)]
if len(employment_status_choices) > 0:
    df_filtered = df_filtered[df_filtered['employment_status'].isin(employment_status_choices)]
if len(income_choices) > 0:
    df_filtered = df_filtered[df_filtered['income'].isin(income_choices)]
if len(education_choices) > 0:
    df_filtered = df_filtered[df_filtered['education'].isin(education_choices)]
if len(location_type_choices) > 0:
    df_filtered = df_filtered[df_filtered['location_type'].isin(location_type_choices)]
if len(household_size_choices) > 0:
    df_filtered = df_filtered[df_filtered['household_size'].isin(household_size_choices)]
df_filtered = df_filtered[(df_filtered['health_consciousness']>= health_consciousness_range[0]) & (df_filtered['health_consciousness']<= health_consciousness_range[1])]
if len(exercise_frequency_choices) > 0:
    df_filtered = df_filtered[df_filtered['exercise_frequency'].isin(exercise_frequency_choices)]
if len(fitness_goal_choices) > 0:
    df_filtered = df_filtered[df_filtered['fitness_goal'].isin(fitness_goal_choices)]
df_filtered = df_filtered[(df_filtered['hydration_importance']>= hydration_importance_range[0]) & (df_filtered['hydration_importance']<= hydration_importance_range[1])]
if len(daily_water_intake_choices) > 0:
    df_filtered = df_filtered[df_filtered['daily_water_intake'].isin(daily_water_intake_choices)]
if len(bottle_type_choices) > 0:
    df_filtered = df_filtered[df_filtered['bottle_type'].isin(bottle_type_choices)]
if len(monthly_beverage_spend_choices) > 0:
    df_filtered = df_filtered[df_filtered['monthly_beverage_spend'].isin(monthly_beverage_spend_choices)]
df_filtered = df_filtered[(df_filtered['interest_level']>= interest_level_range[0]) & (df_filtered['interest_level']<= interest_level_range[1])]
if len(purchase_likelihood_choices) > 0:
    df_filtered = df_filtered[df_filtered['purchase_likelihood'].isin(purchase_likelihood_choices)]
df_filtered = df_filtered[(df_filtered['willingness_to_pay_continuous']>= willingness_to_pay_continuous_range[0]) & (df_filtered['willingness_to_pay_continuous']<= willingness_to_pay_continuous_range[1])]
if len(willingness_to_pay_category_choices) > 0:
    df_filtered = df_filtered[df_filtered['willingness_to_pay_category'].isin(willingness_to_pay_category_choices)]
if len(weekly_usage_choices) > 0:
    df_filtered = df_filtered[df_filtered['weekly_usage'].isin(weekly_usage_choices)]
if len(purchase_preference_choices) > 0:
    df_filtered = df_filtered[df_filtered['purchase_preference'].isin(purchase_preference_choices)]
df_filtered = df_filtered[(df_filtered['sustainability_importance']>= sustainability_importance_range[0]) & (df_filtered['sustainability_importance']<= sustainability_importance_range[1])]
df_filtered = df_filtered[(df_filtered['early_adopter_score']>= early_adopter_score_range[0]) & (df_filtered['early_adopter_score']<= early_adopter_score_range[1])]
df_filtered = df_filtered[(df_filtered['premium_willingness_score']>= premium_willingness_score_range[0]) & (df_filtered['premium_willingness_score']<= premium_willingness_score_range[1])]
df_filtered = df_filtered[(df_filtered['health_condition_Diabetes (Type 1 or 2)']>= health_condition_Diabetes (Type 1 or 2)_range[0]) & (df_filtered['health_condition_Diabetes (Type 1 or 2)']<= health_condition_Diabetes (Type 1 or 2)_range[1])]
df_filtered = df_filtered[(df_filtered['health_condition_High blood pressure']>= health_condition_High blood pressure_range[0]) & (df_filtered['health_condition_High blood pressure']<= health_condition_High blood pressure_range[1])]
df_filtered = df_filtered[(df_filtered['health_condition_High cholesterol']>= health_condition_High cholesterol_range[0]) & (df_filtered['health_condition_High cholesterol']<= health_condition_High cholesterol_range[1])]
df_filtered = df_filtered[(df_filtered['health_condition_Heart disease']>= health_condition_Heart disease_range[0]) & (df_filtered['health_condition_Heart disease']<= health_condition_Heart disease_range[1])]
df_filtered = df_filtered[(df_filtered['health_condition_Kidney disease']>= health_condition_Kidney disease_range[0]) & (df_filtered['health_condition_Kidney disease']<= health_condition_Kidney disease_range[1])]
df_filtered = df_filtered[(df_filtered['health_condition_Food allergies']>= health_condition_Food allergies_range[0]) & (df_filtered['health_condition_Food allergies']<= health_condition_Food allergies_range[1])]
df_filtered = df_filtered[(df_filtered['health_condition_None of the above']>= health_condition_None of the above_range[0]) & (df_filtered['health_condition_None of the above']<= health_condition_None of the above_range[1])]
df_filtered = df_filtered[(df_filtered['barrier_I forget to drink']>= barrier_I forget to drink_range[0]) & (df_filtered['barrier_I forget to drink']<= barrier_I forget to drink_range[1])]
df_filtered = df_filtered[(df_filtered['barrier_Plain water is boring/tasteless']>= barrier_Plain water is boring/tasteless_range[0]) & (df_filtered['barrier_Plain water is boring/tasteless']<= barrier_Plain water is boring/tasteless_range[1])]
df_filtered = df_filtered[(df_filtered['barrier_I dont feel thirsty']>= barrier_I dont feel thirsty_range[0]) & (df_filtered['barrier_I dont feel thirsty']<= barrier_I dont feel thirsty_range[1])]
df_filtered = df_filtered[(df_filtered['barrier_Its inconvenient to carry water']>= barrier_Its inconvenient to carry water_range[0]) & (df_filtered['barrier_Its inconvenient to carry water']<= barrier_Its inconvenient to carry water_range[1])]
df_filtered = df_filtered[(df_filtered['barrier_I prefer other beverages']>= barrier_I prefer other beverages_range[0]) & (df_filtered['barrier_I prefer other beverages']<= barrier_I prefer other beverages_range[1])]
df_filtered = df_filtered[(df_filtered['barrier_Health reasons (frequent bathroom trips, etc.)']>= barrier_Health reasons (frequent bathroom trips, etc.)_range[0]) & (df_filtered['barrier_Health reasons (frequent bathroom trips, etc.)']<= barrier_Health reasons (frequent bathroom trips, etc.)_range[1])]
df_filtered = df_filtered[(df_filtered['barrier_Nothing prevents me']>= barrier_Nothing prevents me_range[0]) & (df_filtered['barrier_Nothing prevents me']<= barrier_Nothing prevents me_range[1])]
df_filtered = df_filtered[(df_filtered['consume_location_At home']>= consume_location_At home_range[0]) & (df_filtered['consume_location_At home']<= consume_location_At home_range[1])]
df_filtered = df_filtered[(df_filtered['consume_location_At work/school']>= consume_location_At work/school_range[0]) & (df_filtered['consume_location_At work/school']<= consume_location_At work/school_range[1])]
df_filtered = df_filtered[(df_filtered['consume_location_At the gym']>= consume_location_At the gym_range[0]) & (df_filtered['consume_location_At the gym']<= consume_location_At the gym_range[1])]
df_filtered = df_filtered[(df_filtered['consume_location_During commute']>= consume_location_During commute_range[0]) & (df_filtered['consume_location_During commute']<= consume_location_During commute_range[1])]
df_filtered = df_filtered[(df_filtered['consume_location_Restaurants/cafes']>= consume_location_Restaurants/cafes_range[0]) & (df_filtered['consume_location_Restaurants/cafes']<= consume_location_Restaurants/cafes_range[1])]
df_filtered = df_filtered[(df_filtered['consume_location_Outdoor activities']>= consume_location_Outdoor activities_range[0]) & (df_filtered['consume_location_Outdoor activities']<= consume_location_Outdoor activities_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Plain water']>= beverage_Plain water_range[0]) & (df_filtered['beverage_Plain water']<= beverage_Plain water_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Flavored water (e.g., LaCroix, Hint)']>= beverage_Flavored water (e.g., LaCroix, Hint)_range[0]) & (df_filtered['beverage_Flavored water (e.g., LaCroix, Hint)']<= beverage_Flavored water (e.g., LaCroix, Hint)_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Sports drinks (e.g., Gatorade, Powerade)']>= beverage_Sports drinks (e.g., Gatorade, Powerade)_range[0]) & (df_filtered['beverage_Sports drinks (e.g., Gatorade, Powerade)']<= beverage_Sports drinks (e.g., Gatorade, Powerade)_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Energy drinks (e.g., Red Bull, Monster)']>= beverage_Energy drinks (e.g., Red Bull, Monster)_range[0]) & (df_filtered['beverage_Energy drinks (e.g., Red Bull, Monster)']<= beverage_Energy drinks (e.g., Red Bull, Monster)_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Enhanced water (e.g., Vitaminwater, Smartwater)']>= beverage_Enhanced water (e.g., Vitaminwater, Smartwater)_range[0]) & (df_filtered['beverage_Enhanced water (e.g., Vitaminwater, Smartwater)']<= beverage_Enhanced water (e.g., Vitaminwater, Smartwater)_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Coffee']>= beverage_Coffee_range[0]) & (df_filtered['beverage_Coffee']<= beverage_Coffee_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Tea']>= beverage_Tea_range[0]) & (df_filtered['beverage_Tea']<= beverage_Tea_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Soda/soft drinks']>= beverage_Soda/soft drinks_range[0]) & (df_filtered['beverage_Soda/soft drinks']<= beverage_Soda/soft drinks_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Juice']>= beverage_Juice_range[0]) & (df_filtered['beverage_Juice']<= beverage_Juice_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Protein shakes']>= beverage_Protein shakes_range[0]) & (df_filtered['beverage_Protein shakes']<= beverage_Protein shakes_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Pre-workout drinks']>= beverage_Pre-workout drinks_range[0]) & (df_filtered['beverage_Pre-workout drinks']<= beverage_Pre-workout drinks_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Electrolyte tablets/powders (e.g., Nuun, Liquid I.V.)']>= beverage_Electrolyte tablets/powders (e.g., Nuun, Liquid I.V.)_range[0]) & (df_filtered['beverage_Electrolyte tablets/powders (e.g., Nuun, Liquid I.V.)']<= beverage_Electrolyte tablets/powders (e.g., Nuun, Liquid I.V.)_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Squeeze flavor enhancers (e.g., MiO, Crystal Light)']>= beverage_Squeeze flavor enhancers (e.g., MiO, Crystal Light)_range[0]) & (df_filtered['beverage_Squeeze flavor enhancers (e.g., MiO, Crystal Light)']<= beverage_Squeeze flavor enhancers (e.g., MiO, Crystal Light)_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Kombucha']>= beverage_Kombucha_range[0]) & (df_filtered['beverage_Kombucha']<= beverage_Kombucha_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Coconut water']>= beverage_Coconut water_range[0]) & (df_filtered['beverage_Coconut water']<= beverage_Coconut water_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Taste/flavor variety']>= priority_Taste/flavor variety_range[0]) & (df_filtered['priority_Taste/flavor variety']<= priority_Taste/flavor variety_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Nutritional benefits (vitamins, minerals)']>= priority_Nutritional benefits (vitamins, minerals)_range[0]) & (df_filtered['priority_Nutritional benefits (vitamins, minerals)']<= priority_Nutritional benefits (vitamins, minerals)_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Low/zero sugar']>= priority_Low/zero sugar_range[0]) & (df_filtered['priority_Low/zero sugar']<= priority_Low/zero sugar_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Natural ingredients']>= priority_Natural ingredients_range[0]) & (df_filtered['priority_Natural ingredients']<= priority_Natural ingredients_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Caffeine content']>= priority_Caffeine content_range[0]) & (df_filtered['priority_Caffeine content']<= priority_Caffeine content_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Electrolytes']>= priority_Electrolytes_range[0]) & (df_filtered['priority_Electrolytes']<= priority_Electrolytes_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Protein content']>= priority_Protein content_range[0]) & (df_filtered['priority_Protein content']<= priority_Protein content_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Convenience/portability']>= priority_Convenience/portability_range[0]) & (df_filtered['priority_Convenience/portability']<= priority_Convenience/portability_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Price/affordability']>= priority_Price/affordability_range[0]) & (df_filtered['priority_Price/affordability']<= priority_Price/affordability_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Brand reputation']>= priority_Brand reputation_range[0]) & (df_filtered['priority_Brand reputation']<= priority_Brand reputation_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Environmental sustainability']>= priority_Environmental sustainability_range[0]) & (df_filtered['priority_Environmental sustainability']<= priority_Environmental sustainability_range[1])]
df_filtered = df_filtered[(df_filtered['priority_No artificial ingredients']>= priority_No artificial ingredients_range[0]) & (df_filtered['priority_No artificial ingredients']<= priority_No artificial ingredients_range[1])]
df_filtered = df_filtered[(df_filtered['used_product_Cirkul (flavor cartridge bottle)']>= used_product_Cirkul (flavor cartridge bottle)_range[0]) & (df_filtered['used_product_Cirkul (flavor cartridge bottle)']<= used_product_Cirkul (flavor cartridge bottle)_range[1])]
df_filtered = df_filtered[(df_filtered['used_product_Air Up (scent-based bottle)']>= used_product_Air Up (scent-based bottle)_range[0]) & (df_filtered['used_product_Air Up (scent-based bottle)']<= used_product_Air Up (scent-based bottle)_range[1])]
df_filtered = df_filtered[(df_filtered['used_product_MiO/liquid flavor drops']>= used_product_MiO/liquid flavor drops_range[0]) & (df_filtered['used_product_MiO/liquid flavor drops']<= used_product_MiO/liquid flavor drops_range[1])]
df_filtered = df_filtered[(df_filtered['used_product_Dissolvable tablets (Nuun, Liquid I.V.)']>= used_product_Dissolvable tablets (Nuun, Liquid I.V.)_range[0]) & (df_filtered['used_product_Dissolvable tablets (Nuun, Liquid I.V.)']<= used_product_Dissolvable tablets (Nuun, Liquid I.V.)_range[1])]
df_filtered = df_filtered[(df_filtered['used_product_Powder packets (Crystal Light, etc.)']>= used_product_Powder packets (Crystal Light, etc.)_range[0]) & (df_filtered['used_product_Powder packets (Crystal Light, etc.)']<= used_product_Powder packets (Crystal Light, etc.)_range[1])]
df_filtered = df_filtered[(df_filtered['used_product_Pre-flavored bottled water']>= used_product_Pre-flavored bottled water_range[0]) & (df_filtered['used_product_Pre-flavored bottled water']<= used_product_Pre-flavored bottled water_range[1])]
df_filtered = df_filtered[(df_filtered['used_product_None of the above']>= used_product_None of the above_range[0]) & (df_filtered['used_product_None of the above']<= used_product_None of the above_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_Convenience/ease of use']>= appealing_Convenience/ease of use_range[0]) & (df_filtered['appealing_Convenience/ease of use']<= appealing_Convenience/ease of use_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_Portion control']>= appealing_Portion control_range[0]) & (df_filtered['appealing_Portion control']<= appealing_Portion control_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_No mixing required']>= appealing_No mixing required_range[0]) & (df_filtered['appealing_No mixing required']<= appealing_No mixing required_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_Flavor variety']>= appealing_Flavor variety_range[0]) & (df_filtered['appealing_Flavor variety']<= appealing_Flavor variety_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_Nutritional customization']>= appealing_Nutritional customization_range[0]) & (df_filtered['appealing_Nutritional customization']<= appealing_Nutritional customization_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_Portability']>= appealing_Portability_range[0]) & (df_filtered['appealing_Portability']<= appealing_Portability_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_Hygiene (single-use)']>= appealing_Hygiene (single-use)_range[0]) & (df_filtered['appealing_Hygiene (single-use)']<= appealing_Hygiene (single-use)_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_Helps me drink more water']>= appealing_Helps me drink more water_range[0]) & (df_filtered['appealing_Helps me drink more water']<= appealing_Helps me drink more water_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_Health benefits']>= appealing_Health benefits_range[0]) & (df_filtered['appealing_Health benefits']<= appealing_Health benefits_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_Nothing appeals to me']>= appealing_Nothing appeals to me_range[0]) & (df_filtered['appealing_Nothing appeals to me']<= appealing_Nothing appeals to me_range[1])]
df_filtered = df_filtered[(df_filtered['concern_Price/cost per use']>= concern_Price/cost per use_range[0]) & (df_filtered['concern_Price/cost per use']<= concern_Price/cost per use_range[1])]
df_filtered = df_filtered[(df_filtered['concern_Environmental waste (single-use plastic)']>= concern_Environmental waste (single-use plastic)_range[0]) & (df_filtered['concern_Environmental waste (single-use plastic)']<= concern_Environmental waste (single-use plastic)_range[1])]
df_filtered = df_filtered[(df_filtered['concern_Powder might not dissolve well']>= concern_Powder might not dissolve well_range[0]) & (df_filtered['concern_Powder might not dissolve well']<= concern_Powder might not dissolve well_range[1])]
df_filtered = df_filtered[(df_filtered['concern_Artificial ingredients']>= concern_Artificial ingredients_range[0]) & (df_filtered['concern_Artificial ingredients']<= concern_Artificial ingredients_range[1])]
df_filtered = df_filtered[(df_filtered['concern_Limited flavor options']>= concern_Limited flavor options_range[0]) & (df_filtered['concern_Limited flavor options']<= concern_Limited flavor options_range[1])]
df_filtered = df_filtered[(df_filtered['concern_Not compatible with my bottle']>= concern_Not compatible with my bottle_range[0]) & (df_filtered['concern_Not compatible with my bottle']<= concern_Not compatible with my bottle_range[1])]
df_filtered = df_filtered[(df_filtered['concern_Prefer other methods']>= concern_Prefer other methods_range[0]) & (df_filtered['concern_Prefer other methods']<= concern_Prefer other methods_range[1])]
df_filtered = df_filtered[(df_filtered['concern_No concerns']>= concern_No concerns_range[0]) & (df_filtered['concern_No concerns']<= concern_No concerns_range[1])]
if len(price_15_perception_choices) > 0:
    df_filtered = df_filtered[df_filtered['price_15_perception'].isin(price_15_perception_choices)]
if len(price_25_perception_choices) > 0:
    df_filtered = df_filtered[df_filtered['price_25_perception'].isin(price_25_perception_choices)]
if len(price_35_perception_choices) > 0:
    df_filtered = df_filtered[df_filtered['price_35_perception'].isin(price_35_perception_choices)]
if len(price_45_perception_choices) > 0:
    df_filtered = df_filtered[df_filtered['price_45_perception'].isin(price_45_perception_choices)]
if len(price_60_perception_choices) > 0:
    df_filtered = df_filtered[df_filtered['price_60_perception'].isin(price_60_perception_choices)]
df_filtered = df_filtered[(df_filtered['flavor_Citrus (lemon, lime, orange)']>= flavor_Citrus (lemon, lime, orange)_range[0]) & (df_filtered['flavor_Citrus (lemon, lime, orange)']<= flavor_Citrus (lemon, lime, orange)_range[1])]
df_filtered = df_filtered[(df_filtered['flavor_Berry (strawberry, blueberry, raspberry)']>= flavor_Berry (strawberry, blueberry, raspberry)_range[0]) & (df_filtered['flavor_Berry (strawberry, blueberry, raspberry)']<= flavor_Berry (strawberry, blueberry, raspberry)_range[1])]
df_filtered = df_filtered[(df_filtered['flavor_Tropical (mango, pineapple, coconut)']>= flavor_Tropical (mango, pineapple, coconut)_range[0]) & (df_filtered['flavor_Tropical (mango, pineapple, coconut)']<= flavor_Tropical (mango, pineapple, coconut)_range[1])]
df_filtered = df_filtered[(df_filtered['flavor_Mint/herbal']>= flavor_Mint/herbal_range[0]) & (df_filtered['flavor_Mint/herbal']<= flavor_Mint/herbal_range[1])]
df_filtered = df_filtered[(df_filtered['flavor_Green tea/matcha']>= flavor_Green tea/matcha_range[0]) & (df_filtered['flavor_Green tea/matcha']<= flavor_Green tea/matcha_range[1])]
df_filtered = df_filtered[(df_filtered['flavor_Coffee-flavored']>= flavor_Coffee-flavored_range[0]) & (df_filtered['flavor_Coffee-flavored']<= flavor_Coffee-flavored_range[1])]
df_filtered = df_filtered[(df_filtered['flavor_Neutral/unflavored (just nutrients)']>= flavor_Neutral/unflavored (just nutrients)_range[0]) & (df_filtered['flavor_Neutral/unflavored (just nutrients)']<= flavor_Neutral/unflavored (just nutrients)_range[1])]
df_filtered = df_filtered[(df_filtered['flavor_Sour/tangy']>= flavor_Sour/tangy_range[0]) & (df_filtered['flavor_Sour/tangy']<= flavor_Sour/tangy_range[1])]
df_filtered = df_filtered[(df_filtered['flavor_Sweet/dessert-inspired']>= flavor_Sweet/dessert-inspired_range[0]) & (df_filtered['flavor_Sweet/dessert-inspired']<= flavor_Sweet/dessert-inspired_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_Energy boost (caffeine, B-vitamins)']>= benefit_rank_Energy boost (caffeine, B-vitamins)_range[0]) & (df_filtered['benefit_rank_Energy boost (caffeine, B-vitamins)']<= benefit_rank_Energy boost (caffeine, B-vitamins)_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_Hydration/electrolytes']>= benefit_rank_Hydration/electrolytes_range[0]) & (df_filtered['benefit_rank_Hydration/electrolytes']<= benefit_rank_Hydration/electrolytes_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_Immunity support (Vitamin C, zinc)']>= benefit_rank_Immunity support (Vitamin C, zinc)_range[0]) & (df_filtered['benefit_rank_Immunity support (Vitamin C, zinc)']<= benefit_rank_Immunity support (Vitamin C, zinc)_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_Focus/mental clarity']>= benefit_rank_Focus/mental clarity_range[0]) & (df_filtered['benefit_rank_Focus/mental clarity']<= benefit_rank_Focus/mental clarity_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_Recovery (post-workout)']>= benefit_rank_Recovery (post-workout)_range[0]) & (df_filtered['benefit_rank_Recovery (post-workout)']<= benefit_rank_Recovery (post-workout)_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_Digestive health (probiotics)']>= benefit_rank_Digestive health (probiotics)_range[0]) & (df_filtered['benefit_rank_Digestive health (probiotics)']<= benefit_rank_Digestive health (probiotics)_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_Antioxidants']>= benefit_rank_Antioxidants_range[0]) & (df_filtered['benefit_rank_Antioxidants']<= benefit_rank_Antioxidants_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_Skin health (collagen, biotin)']>= benefit_rank_Skin health (collagen, biotin)_range[0]) & (df_filtered['benefit_rank_Skin health (collagen, biotin)']<= benefit_rank_Skin health (collagen, biotin)_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_Weight management']>= benefit_rank_Weight management_range[0]) & (df_filtered['benefit_rank_Weight management']<= benefit_rank_Weight management_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_General vitamins/minerals']>= benefit_rank_General vitamins/minerals_range[0]) & (df_filtered['benefit_rank_General vitamins/minerals']<= benefit_rank_General vitamins/minerals_range[1])]
df_filtered = df_filtered[(df_filtered['specialized_Sugar-free (for diabetics)']>= specialized_Sugar-free (for diabetics)_range[0]) & (df_filtered['specialized_Sugar-free (for diabetics)']<= specialized_Sugar-free (for diabetics)_range[1])]
df_filtered = df_filtered[(df_filtered['specialized_Keto-friendly']>= specialized_Keto-friendly_range[0]) & (df_filtered['specialized_Keto-friendly']<= specialized_Keto-friendly_range[1])]
df_filtered = df_filtered[(df_filtered['specialized_Vegan']>= specialized_Vegan_range[0]) & (df_filtered['specialized_Vegan']<= specialized_Vegan_range[1])]
df_filtered = df_filtered[(df_filtered['specialized_Organic/natural only']>= specialized_Organic/natural only_range[0]) & (df_filtered['specialized_Organic/natural only']<= specialized_Organic/natural only_range[1])]
df_filtered = df_filtered[(df_filtered['specialized_Allergen-free (gluten, dairy, soy)']>= specialized_Allergen-free (gluten, dairy, soy)_range[0]) & (df_filtered['specialized_Allergen-free (gluten, dairy, soy)']<= specialized_Allergen-free (gluten, dairy, soy)_range[1])]
df_filtered = df_filtered[(df_filtered['specialized_Kid-friendly formulas']>= specialized_Kid-friendly formulas_range[0]) & (df_filtered['specialized_Kid-friendly formulas']<= specialized_Kid-friendly formulas_range[1])]
df_filtered = df_filtered[(df_filtered['specialized_Senior-optimized (bone health, etc.)']>= specialized_Senior-optimized (bone health, etc.)_range[0]) & (df_filtered['specialized_Senior-optimized (bone health, etc.)']<= specialized_Senior-optimized (bone health, etc.)_range[1])]
df_filtered = df_filtered[(df_filtered['specialized_Athletic performance']>= specialized_Athletic performance_range[0]) & (df_filtered['specialized_Athletic performance']<= specialized_Athletic performance_range[1])]
df_filtered = df_filtered[(df_filtered['specialized_None - prefer standard versions']>= specialized_None - prefer standard versions_range[0]) & (df_filtered['specialized_None - prefer standard versions']<= specialized_None - prefer standard versions_range[1])]
df_filtered = df_filtered[(df_filtered['discovery_Social media (Instagram, TikTok, Facebook)']>= discovery_Social media (Instagram, TikTok, Facebook)_range[0]) & (df_filtered['discovery_Social media (Instagram, TikTok, Facebook)']<= discovery_Social media (Instagram, TikTok, Facebook)_range[1])]
df_filtered = df_filtered[(df_filtered['discovery_Friends/family recommendations']>= discovery_Friends/family recommendations_range[0]) & (df_filtered['discovery_Friends/family recommendations']<= discovery_Friends/family recommendations_range[1])]
df_filtered = df_filtered[(df_filtered['discovery_Health/fitness influencers']>= discovery_Health/fitness influencers_range[0]) & (df_filtered['discovery_Health/fitness influencers']<= discovery_Health/fitness influencers_range[1])]
df_filtered = df_filtered[(df_filtered['discovery_Online reviews']>= discovery_Online reviews_range[0]) & (df_filtered['discovery_Online reviews']<= discovery_Online reviews_range[1])]
df_filtered = df_filtered[(df_filtered['discovery_In-store displays']>= discovery_In-store displays_range[0]) & (df_filtered['discovery_In-store displays']<= discovery_In-store displays_range[1])]
df_filtered = df_filtered[(df_filtered['discovery_TV/online ads']>= discovery_TV/online ads_range[0]) & (df_filtered['discovery_TV/online ads']<= discovery_TV/online ads_range[1])]
df_filtered = df_filtered[(df_filtered['discovery_Health blogs/websites']>= discovery_Health blogs/websites_range[0]) & (df_filtered['discovery_Health blogs/websites']<= discovery_Health blogs/websites_range[1])]
df_filtered = df_filtered[(df_filtered['discovery_Nutritionist/doctor recommendation']>= discovery_Nutritionist/doctor recommendation_range[0]) & (df_filtered['discovery_Nutritionist/doctor recommendation']<= discovery_Nutritionist/doctor recommendation_range[1])]

st.write(f"Filtered rows: {len(df_filtered)}")
st.dataframe(df_filtered.head(50))

tab1, tab2, tab3, tab4 = st.tabs(["Classification","Association Rules","Clustering","Regression"])

with tab1:
    st.header("Classification")
    st.write("Select target (categorical) and features.")
    cat_cols = [c for c in df_filtered.columns if (df_filtered[c].nunique()<=20) and (df_filtered[c].dtype=='object' or df_filtered[c].dtype.name=='category')]
    st.write("Candidate categorical targets (<=20 unique):", cat_cols)
    target = st.selectbox("Target column (classification)", options=cat_cols)
    features = st.multiselect("Features (use numeric and categorical)", options=[c for c in df_filtered.columns if c!=target], default=[c for c in df_filtered.columns if c!=target][:5])
    if st.button("Run classification"):
        sub = df_filtered[features+[target]].dropna()
        X = sub[features].copy()
        y = sub[target].copy()
        X_proc = X.copy()
        for col in X_proc.columns:
            if X_proc[col].dtype=='object' or X_proc[col].dtype.name=='category':
                X_proc[col] = LabelEncoder().fit_transform(X_proc[col].astype(str))
            else:
                X_proc[col] = X_proc[col].astype(float).fillna(X_proc[col].mean())
        y_enc = LabelEncoder().fit_transform(y.astype(str))
        X_train, X_test, y_train, y_test = train_test_split(X_proc, y_enc, test_size=0.25, random_state=42, stratify=y_enc)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.write(f"Accuracy: {acc:.3f}")
        st.text(classification_report(y_test, preds))

with tab2:
    st.header("Association Rule Mining")
    st.write("Prepare transactional data: select columns to treat as one-hot encoded items (categorical).")
    cat_cols = [c for c in df_filtered.columns if df_filtered[c].nunique()<=50]
    items = st.multiselect("Columns to include as items", options=cat_cols, default=cat_cols[:5])
    min_support = st.slider("Min support", 0.01, 0.5, 0.05)
    min_conf = st.slider("Min confidence", 0.1, 1.0, 0.6)
    if st.button("Run association rules"):
        trans = df_filtered[items].astype(str).fillna('NA')
        one_hot = pd.get_dummies(trans.apply(lambda row: row.astype(str), axis=1).stack()).groupby(level=0).sum()
        frequent = apriori(one_hot, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
        st.write("Frequent itemsets:", frequent.sort_values('support', ascending=False).head(20))
        st.write("Rules:", rules[['antecedents','consequents','support','confidence','lift']].sort_values('confidence', ascending=False).head(20))

with tab3:
    st.header("Clustering (KMeans)")
    numeric_cols = [c for c in df_filtered.columns if pd.api.types.is_numeric_dtype(df_filtered[c])]
    st.write("Numeric columns:", numeric_cols)
    k = st.slider("Number of clusters (k)", 2, 10, 3)
    cols_for_k = st.multiselect("Columns to use for clustering", options=numeric_cols, default=numeric_cols[:2] if len(numeric_cols)>=2 else numeric_cols)
    if st.button("Run clustering"):
        if len(cols_for_k)<1:
            st.error("Pick at least one numeric column.")
        else:
            X = df_filtered[cols_for_k].dropna()
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(Xs)
            Xv = pd.DataFrame(Xs, columns=cols_for_k)
            Xv['cluster'] = labels
            st.write(Xv.groupby('cluster').mean())
            if len(cols_for_k)==2:
                chart = alt.Chart(Xv).mark_circle(size=60).encode(
                    x=cols_for_k[0], y=cols_for_k[1], color='cluster:N', tooltip=list(Xv.columns)
                ).interactive()
                st.altair_chart(chart, use_container_width=True)

with tab4:
    st.header("Regression")
    st.write("Select target (numeric) and features.")
    num_cols = [c for c in df_filtered.columns if pd.api.types.is_numeric_dtype(df_filtered[c])]
    st.write("Numeric columns:", num_cols)
    target_r = st.selectbox("Target column (regression)", options=num_cols)
    features_r = st.multiselect("Features (numeric or encoded)", options=[c for c in df_filtered.columns if c!=target_r], default=[c for c in df_filtered.columns if c!=target_r][:5])
    if st.button("Run regression"):
        sub = df_filtered[features_r+[target_r]].dropna()
        X = sub[features_r].copy()
        y = sub[target_r].astype(float).copy()
        for col in X.columns:
            if X[col].dtype=='object' or X[col].dtype.name=='category':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        st.write(f"MSE: {mse:.3f}, R²: {r2:.3f}")
        res_df = pd.DataFrame({'actual': y_test, 'predicted': preds})
        st.line_chart(res_df.reset_index(drop=True))
