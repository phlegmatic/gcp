-- Mart: `demand_features` -- the single table consumed by both KFP pipelines.
-- Materialized as a table so training/drift steps do only cheap SELECT * scans.
{{ config(materialized='table') }}

with base as (
    select ds, demand from {{ ref('stg_sales') }}
),

featured as (
    select
        ds,
        demand,
        lag(demand, 1)  over (order by ds) as lag_1,
        lag(demand, 7)  over (order by ds) as lag_7,
        lag(demand, 14) over (order by ds) as lag_14,
        avg(demand) over (
            order by ds rows between 7 preceding and 1 preceding
        ) as roll_mean_7,
        avg(demand) over (
            order by ds rows between 28 preceding and 1 preceding
        ) as roll_mean_28,
        extract(dayofweek from ds) as dayofweek,
        extract(month from ds)     as month
    from base
)

select *
from featured
where lag_14 is not null  -- drop warmup rows so downstream has no NaNs
