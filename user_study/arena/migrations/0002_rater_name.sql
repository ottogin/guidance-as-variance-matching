-- Optional self-reported name or initials, typed on the intro screen. Never sent back to the page.
ALTER TABLE raters ADD COLUMN name TEXT CHECK (name IS NULL OR length(name) <= 60);

-- One row per rater for the organiser: who, how much, when.
CREATE VIEW rater_summary AS
  SELECT r.id, r.name, r.source,
         count(v.seq)                                   AS votes,
         sum(v.result = 't')                            AS ties,
         r.skips,
         datetime(r.created_at / 1000, 'unixepoch')     AS first_seen_utc,
         datetime(r.last_seen  / 1000, 'unixepoch')     AS last_seen_utc,
         round(avg(v.rt_ds) / 10.0, 1)                  AS mean_seconds
  FROM raters r LEFT JOIN votes v ON v.rater = r.id
  GROUP BY r.id ORDER BY votes DESC;
