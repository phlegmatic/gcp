-- Staging: clean + standardize raw sales into a daily grain.
with source as (
    select * from {{ source('demand_raw', 'sales') }}
)
select
    cast(sale_date as date)         as ds,
    cast(units_sold as float64)     as demand
from source
where units_sold is not null
