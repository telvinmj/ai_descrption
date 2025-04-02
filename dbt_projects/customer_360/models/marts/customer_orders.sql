{{ config(materialized='table') }}

WITH customers AS (
    SELECT
        customer_id,
        first_name,
        last_name,
        email,
        customer_segment
    FROM {{ ref('stg_customers') }}
),

retail_sales AS (
    SELECT
        customer_id,
        sale_id,
        sale_date,
        CAST(total_amount AS DECIMAL(10,2)) as total_amount
    FROM retail.raw_sales
),

customer_orders AS (
    SELECT
        c.customer_id,
        c.first_name,
        c.last_name,
        c.email,
        c.customer_segment,
        COUNT(s.sale_id) AS order_count,
        SUM(s.total_amount) AS total_spend,
        MIN(s.sale_date) AS first_order_date,
        MAX(s.sale_date) AS most_recent_order_date
    FROM customers c
    LEFT JOIN retail_sales s ON c.customer_id = s.customer_id
    GROUP BY 1, 2, 3, 4, 5
)

SELECT * FROM customer_orders 