{{ config(materialized='table') }}

WITH customer_base AS (
    SELECT
        c.*,
        cp.preferred_category,
        cp.communication_channel
    FROM {{ ref('stg_customers') }} c
    LEFT JOIN {{ ref('stg_customer_preferences') }} cp ON c.customer_id = cp.customer_id
),

customer_feedback_agg AS (
    SELECT
        customer_id,
        AVG(rating) as avg_rating,
        COUNT(*) as total_feedback_count
    FROM {{ ref('stg_customer_feedback') }}
    GROUP BY customer_id
),

final_profile AS (
    SELECT
        cb.*,
        COALESCE(cf.avg_rating, 0) as average_rating,
        COALESCE(cf.total_feedback_count, 0) as feedback_count
    FROM customer_base cb
    LEFT JOIN customer_feedback_agg cf ON cb.customer_id = cf.customer_id
)

SELECT * FROM final_profile 