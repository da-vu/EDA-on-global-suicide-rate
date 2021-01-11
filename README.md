# ECE143_project_ EDA-on-global-suicide-rate

Fall 2020 UCSD ECE143

The scope of the project is the demographic analysis of suicide all over the world.

The files include the codes and the outputs graphs of the project.


## Team members:
- Dan Vu
- Dylan Perlson
- Payam Khorramshahi
- Zhengdong Gao

## Motivations:
What characteristics and attributes are correlated with a higher risk for suicide

## Objectives: Answer the questions below
- How do different factors affect suicide rates?
- How does each group contribute towards changes in suicide numbers?
- Which populations are at highest risk of suicide?
- What correlations are there between population and suicide numbers?

## Summary
- When analyzing the TOTAL count of suicides, the age group 35-54 has the highest contribution
- When analyzing the suicide rates, the age group 75+ contribute a much higher percentage
- As for the gender, men tend to have a higher suicide rate than women
- Out of all the continents, Europe has had both a higher suicide rate AND a more dramatic change in suicide rates over the years
- In some countries, suicide rate is negatively correlated with population
- Economy has influence on the suicide rate

## Dataset
- https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016
- https://www.kaggle.com/szamil/who-suicide-statistics

## File Structure

```
Final.py
|
+----raw_data
|
+----customfunction
    |
    +----processed_data
    |       |   remove_HDI
    |       |   change_gdp_to_ints
    |       |   rename_suicide_rate
    |       |   remove_2016
    |       |   add_conts
    +----main_function
    |       |   suiciderate_gender_year
    |       |   suicidecount_gender_year
    |       |   suiciderate_gdp_gender
    |       |   suicidecount_gdp_gender
    |       |   suiciderate_age_year
    |       |   suiciderate_age
    |       |   suiciderate_cont_time
    |       |   suiciderate_country_time
    |       |   suicides_cont_avg
    |       |   suicide_gdp
    |       |   barplot
    |       |   categorize
    |       |   suicide_gender_year
    |       |   suicides_gdp_gender
    |       |   countries_suicide_rate
    |       |   suicides_age_year
    |       |   suicides_per100k_age
    |       |   suicides_no_age
    |       |   suicide_by_age_pie
    |       |   suicide_by_year
    |       |   suicide_by_age_range
    |       |   suicide_by_sex
    |       |   suicide_by_age
    |       |   suicide_pearson_population
    |       |   Sui_by_pop
```
## Instructions on running the code

* Python version: Python 3.7.6 64-bit
### Required packages

1. numpy
2. pandas
3. matplotlib
4. seaborn

### Run the code
1. Run the ```Final.py``` to generate all the data from raw csv files in ```./master.csv```   
2. Run the function in customfunctions to get the graphs.
