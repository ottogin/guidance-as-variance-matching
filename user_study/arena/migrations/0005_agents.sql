-- AI raters (source 'agent', or 'claude' for imported AI votes) group by model name in their own
-- namespace, so a model never merges with a human who typed the same name.
DROP VIEW rater_summary;
DROP VIEW rater_person;
CREATE VIEW rater_person AS
  SELECT id, name,
         CASE WHEN source IN ('agent', 'claude') AND name IS NOT NULL AND name <> '' THEN 'agent:' || lower(name)
              WHEN name IS NULL OR name = '' THEN 'id:' || id
              ELSE 'name:' || lower(name) END AS person
  FROM raters;
CREATE VIEW rater_summary AS
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

-- Pairs handed to an agent. An agent can only vote on a task issued to it, so it cannot pick its own pairs.
CREATE TABLE agent_tasks (
  id         TEXT PRIMARY KEY,
  rater      TEXT NOT NULL REFERENCES raters(id),
  pos        INTEGER NOT NULL,
  left_arm   TEXT NOT NULL,
  right_arm  TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  done       INTEGER NOT NULL DEFAULT 0,
  result     TEXT CHECK (result IS NULL OR result IN ('l','r','t')),
  reason     TEXT CHECK (reason IS NULL OR length(reason) <= 1000)
);
CREATE INDEX agent_tasks_open ON agent_tasks(rater, done);
