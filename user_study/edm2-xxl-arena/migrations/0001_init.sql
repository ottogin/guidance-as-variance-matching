-- EDM2 ImageNet-512 pairwise study. One row per judgement; aggregates are views.
-- The final schema of the SD3.5-M arena (its migrations 0001-0007) in one file, with this study's arm codes:
--   n = EDM2-S <- XS-uncond CFG at NVIDIA's FID-tuned w = 1.40, o = the same pair at our variance-matched constant.
-- Arm codes must match build.py's ARMS (build.py checks). Adding arms after deployment needs a new migration that
-- rebuilds `votes` with the wider CHECK (SQLite cannot alter a CHECK in place), as the SD3.5 arena's 0006 did.

CREATE TABLE raters (
  id         TEXT PRIMARY KEY CHECK (length(id) BETWEEN 16 AND 64),  -- random id kept in the rater's browser
  created_at INTEGER NOT NULL,                                       -- unix ms
  last_seen  INTEGER NOT NULL,
  skips      INTEGER NOT NULL DEFAULT 0 CHECK (skips >= 0),
  source     TEXT,                                                   -- 'web' (people) or 'agent' (the rating API)
  name       TEXT CHECK (name IS NULL OR length(name) <= 60)         -- optional name; never sent back to the page
);

CREATE TABLE votes (
  rater      TEXT    NOT NULL REFERENCES raters(id),
  seq        INTEGER NOT NULL CHECK (seq >= 0),       -- per-rater counter: a retried upload cannot double-count
  pos        INTEGER NOT NULL CHECK (pos BETWEEN 0 AND 99),    -- row position (sprite s/NNN.webp)
  left_arm   TEXT    NOT NULL CHECK (left_arm  IN ('a','b','c','d')),
  right_arm  TEXT    NOT NULL CHECK (right_arm IN ('a','b','c','d')),
  result     TEXT    NOT NULL CHECK (result IN ('l','r','t')),  -- left better, right better, tie
  rt_ds      INTEGER CHECK (rt_ds >= 0),                         -- response time, deciseconds
  created_at INTEGER NOT NULL,
  PRIMARY KEY (rater, seq),
  CHECK (left_arm <> right_arm)
) WITHOUT ROWID;

CREATE INDEX votes_by_pos ON votes(pos);

-- Owner settings: 'excluded' -> JSON list of person keys left out of the board; 'aliases' -> person -> display alias.
CREATE TABLE settings (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

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

-- Each judgement seen from both arms' side: (arm, opponent, score) with score 1 / 0.5 / 0.
CREATE VIEW outcomes AS
  SELECT rater, pos, left_arm  AS arm, right_arm AS opp,
         CASE result WHEN 'l' THEN 1.0 WHEN 't' THEN 0.5 ELSE 0.0 END AS score FROM votes
  UNION ALL
  SELECT rater, pos, right_arm AS arm, left_arm  AS opp,
         CASE result WHEN 'r' THEN 1.0 WHEN 't' THEN 0.5 ELSE 0.0 END AS score FROM votes;

-- Per rater and ordered arm pair: games, wins, ties. The page fits Bradley-Terry and bootstraps raters from this.
CREATE VIEW pair_counts AS
  SELECT rater, arm, opp,
         count(*)                        AS games,
         sum(score = 1.0)                AS wins,
         sum(score = 0.5)                AS ties
  FROM outcomes GROUP BY rater, arm, opp;

-- A person is a name: every browser that typed the same name, ignoring case, counts as one person; a browser with no
-- name is its own person. AI raters (source 'agent') group by model name in their own namespace.
CREATE VIEW rater_person AS
  SELECT id, name,
         CASE WHEN source IN ('agent', 'claude') AND name IS NOT NULL AND name <> '' THEN 'agent:' || lower(name)
              WHEN name IS NULL OR name = '' THEN 'id:' || id
              ELSE 'name:' || lower(name) END AS person
  FROM raters;

-- One row per person for the organiser: who, how much, when.
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
