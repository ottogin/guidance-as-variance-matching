-- SD3.5-M seven-arm pairwise study. One row per judgement; aggregates are views.
-- Arm codes: f = const 1.6 (FID-tuned), d = const 2.4 (FD_DINOv2-tuned), o = const 4.14 (ours),
-- s = trace schedule (ours), i / l / v = interval / linear / TV-CFG at 4.5.

CREATE TABLE raters (
  id         TEXT PRIMARY KEY CHECK (length(id) BETWEEN 16 AND 64),  -- random id kept in the rater's browser
  created_at INTEGER NOT NULL,                                       -- unix ms
  last_seen  INTEGER NOT NULL,
  skips      INTEGER NOT NULL DEFAULT 0 CHECK (skips >= 0),
  source     TEXT                                                    -- 'web', or 'claude-db' for votes imported from an earlier copy of the page
);

CREATE TABLE votes (
  rater      TEXT    NOT NULL REFERENCES raters(id),
  seq        INTEGER NOT NULL CHECK (seq >= 0),       -- per-rater counter: a retried upload cannot double-count
  pos        INTEGER NOT NULL CHECK (pos BETWEEN 0 AND 499),   -- caption position (sprite s/NNN.webp)
  left_arm   TEXT    NOT NULL CHECK (left_arm  IN ('f','d','o','s','i','l','v')),
  right_arm  TEXT    NOT NULL CHECK (right_arm IN ('f','d','o','s','i','l','v')),
  result     TEXT    NOT NULL CHECK (result IN ('l','r','t')),  -- left better, right better, tie
  rt_ds      INTEGER CHECK (rt_ds >= 0),                         -- response time, deciseconds
  created_at INTEGER NOT NULL,
  PRIMARY KEY (rater, seq),
  CHECK (left_arm <> right_arm)
) WITHOUT ROWID;

CREATE INDEX votes_by_pos ON votes(pos);

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
