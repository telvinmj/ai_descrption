{{ config(materialized='view') }}

SELECT
    customer_id,
    first_name,
    last_name,
    email,
    signup_date,
    customer_segment
FROM customer.raw_customers 