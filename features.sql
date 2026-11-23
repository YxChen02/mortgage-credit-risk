WITH history AS (
    SELECT *,
        count(*) OVER past AS observed_periods,
        avg(balance_time) OVER past AS mean_balance_3obs,
        avg(LTV_time) OVER past AS mean_ltv_3obs,
        lag(balance_time) OVER sequence AS previous_balance,
        min(CASE WHEN status_time IN (1, 2) THEN time END)
            OVER (PARTITION BY id, orig_time) AS event_period,
        max(time) OVER (PARTITION BY id, orig_time) AS last_seen
    FROM mortgage
    WINDOW sequence AS (PARTITION BY id, orig_time ORDER BY time),
           past AS (PARTITION BY id, orig_time ORDER BY time ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
), labeled AS (
    SELECT *,
        CASE
            WHEN event_period <= time + :horizon THEN
                (SELECT status_time = 1 FROM mortgage AS event
                 WHERE event.id = history.id AND event.orig_time = history.orig_time AND event.time = history.event_period)
            WHEN last_seen >= time + :horizon THEN 0
        END AS target
    FROM history
    WHERE status_time = 0 AND (event_period IS NULL OR event_period > time)
      AND time + :horizon <= :data_end
)
SELECT id, time, orig_time, time - orig_time AS age,
       balance_time, LTV_time, interest_rate_time, FICO_orig_time,
       LTV_orig_time, observed_periods, mean_balance_3obs, mean_ltv_3obs,
       balance_time - previous_balance AS balance_change, target
FROM labeled
