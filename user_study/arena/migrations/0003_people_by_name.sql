-- A person is a name: every browser (rater id) that typed the same name, ignoring case, counts as one person.
-- A browser with no name is its own person. Votes stay keyed by browser id, so uploads remain idempotent.
CREATE VIEW rater_person AS
  SELECT id, name,
         CASE WHEN name IS NULL OR name = '' THEN 'id:' || id ELSE 'name:' || lower(name) END AS person
  FROM raters;

DROP VIEW rater_summary;
CREATE VIEW rater_summary AS                      -- one row per person
  SELECT p.person,
         max(p.name)                                     AS name,
         count(DISTINCT p.id)                            AS browsers,
         count(v.seq)                                    AS votes,
         sum(v.result = 't')                             AS ties,
         datetime(min(r.created_at) / 1000, 'unixepoch') AS first_seen_utc,
         datetime(max(r.last_seen)  / 1000, 'unixepoch') AS last_seen_utc,
         round(avg(v.rt_ds) / 10.0, 1)                   AS mean_seconds
  FROM rater_person p JOIN raters r ON r.id = p.id LEFT JOIN votes v ON v.rater = p.id
  GROUP BY p.person ORDER BY votes DESC;
