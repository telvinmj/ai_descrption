{{ config(materialized='view') }}

SELECT
    preference_id,
    customer_id,
    preferred_category,
    communication_channel,
    last_updated
FROM customer.raw_customer_preferences 