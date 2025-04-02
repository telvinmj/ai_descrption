{{ config(materialized='view') }}

SELECT
    feedback_id,
    customer_id,
    sale_id,
    rating,
    feedback_text,
    feedback_date
FROM customer.raw_customer_feedback 