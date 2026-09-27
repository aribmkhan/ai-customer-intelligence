-- ============================================================
-- NovaMart Customer Intelligence Platform
-- Customer-Level Behavioral Metrics
-- ============================================================

CREATE OR REPLACE TABLE
    `novamart-customer-intelligence.novamart_analytics.customer_metrics`
AS

WITH transaction_details AS (

    SELECT
        t.transaction_id,
        t.customer_id,
        t.product_id,
        t.transaction_date,
        t.quantity,
        t.unit_price,
        t.discount_pct,
        t.order_status,
        p.category,

        -- Revenue after discount
        t.quantity
            * t.unit_price
            * (1 - t.discount_pct / 100)
            AS transaction_revenue

    FROM
        `novamart-customer-intelligence.novamart_analytics.transactions` AS t

    LEFT JOIN
        `novamart-customer-intelligence.novamart_analytics.products` AS p
        ON t.product_id = p.product_id
),


-- ============================================================
-- COMPLETED TRANSACTION METRICS
-- ============================================================

completed_metrics AS (

    SELECT
        customer_id,

        COUNT(DISTINCT transaction_id)
            AS total_orders,

        SUM(transaction_revenue)
            AS total_revenue,

        AVG(transaction_revenue)
            AS average_order_value,

        MIN(transaction_date)
            AS first_purchase_date,

        MAX(transaction_date)
            AS last_purchase_date,

        SUM(quantity)
            AS total_items,

        AVG(discount_pct)
            AS average_discount,

        COUNT(DISTINCT category)
            AS category_count

    FROM transaction_details

    WHERE order_status = 'Completed'

    GROUP BY customer_id
),


-- ============================================================
-- ALL TRANSACTION METRICS
-- Used for return-rate calculation
-- ============================================================

transaction_status_metrics AS (

    SELECT
        customer_id,

        COUNT(*) AS all_transactions,

        COUNTIF(order_status = 'Returned')
            AS returned_transactions

    FROM transaction_details

    GROUP BY customer_id
),


-- ============================================================
-- FINAL CUSTOMER METRICS
-- ============================================================

customer_metrics AS (

    SELECT
        c.customer_id,
        c.signup_date,
        c.age,
        c.gender,
        c.city,
        c.province,
        c.acquisition_channel,
        c.device_type,

        cm.total_orders,

        ROUND(
            cm.total_revenue,
            2
        ) AS total_revenue,

        ROUND(
            cm.average_order_value,
            2
        ) AS average_order_value,

        cm.first_purchase_date,
        cm.last_purchase_date,

        DATE_DIFF(
            DATE '2026-08-31',
            cm.last_purchase_date,
            DAY
        ) AS recency_days,

        cm.total_items,

        ROUND(
            cm.average_discount,
            2
        ) AS average_discount,

        cm.category_count,

        ROUND(
            SAFE_DIVIDE(
                sm.returned_transactions,
                sm.all_transactions
            ),
            4
        ) AS return_rate,

        DATE_DIFF(
            DATE '2026-08-31',
            cm.first_purchase_date,
            DAY
        ) AS customer_tenure_days

    FROM completed_metrics AS cm

    INNER JOIN
        `novamart-customer-intelligence.novamart_analytics.customers` AS c
        ON cm.customer_id = c.customer_id

    LEFT JOIN transaction_status_metrics AS sm
        ON cm.customer_id = sm.customer_id
)


-- ============================================================
-- OUTPUT
-- ============================================================

SELECT *
FROM customer_metrics;