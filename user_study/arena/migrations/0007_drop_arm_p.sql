-- Arm p (the released constant w = 4.5, added by 0006) is withdrawn from the study. Its votes go, and so do
-- the raters whose every vote involved p (all of them web raters who joined through ?focus=p). The votes table is
-- then rebuilt with the seven-arm CHECK of 0001-0005, as 0006 rebuilt it to let p in; the views are recreated unchanged.
CREATE TABLE drop_p_raters AS
  SELECT rater AS id FROM votes GROUP BY rater HAVING sum(left_arm <> 'p' AND right_arm <> 'p') = 0;
DELETE FROM votes WHERE left_arm = 'p' OR right_arm = 'p';
DELETE FROM agent_tasks WHERE left_arm = 'p' OR right_arm = 'p';
DELETE FROM raters WHERE id IN (SELECT id FROM drop_p_raters) AND source = 'web';
DROP TABLE drop_p_raters;

DROP VIEW rater_summary;
DROP VIEW pair_counts;
DROP VIEW outcomes;

CREATE TABLE votes_new (
  rater      TEXT    NOT NULL REFERENCES raters(id),
  seq        INTEGER NOT NULL CHECK (seq >= 0),
  pos        INTEGER NOT NULL CHECK (pos BETWEEN 0 AND 499),
  left_arm   TEXT    NOT NULL CHECK (left_arm  IN ('f','d','o','s','i','l','v')),
  right_arm  TEXT    NOT NULL CHECK (right_arm IN ('f','d','o','s','i','l','v')),
  result     TEXT    NOT NULL CHECK (result IN ('l','r','t')),
  rt_ds      INTEGER CHECK (rt_ds >= 0),
  created_at INTEGER NOT NULL,
  PRIMARY KEY (rater, seq),
  CHECK (left_arm <> right_arm)
) WITHOUT ROWID;
INSERT INTO votes_new (rater, seq, pos, left_arm, right_arm, result, rt_ds, created_at)
  SELECT rater, seq, pos, left_arm, right_arm, result, rt_ds, created_at FROM votes;
DROP TABLE votes;
ALTER TABLE votes_new RENAME TO votes;
CREATE INDEX votes_by_pos ON votes(pos);

CREATE VIEW outcomes AS
  SELECT rater, pos, left_arm  AS arm, right_arm AS opp,
         CASE result WHEN 'l' THEN 1.0 WHEN 't' THEN 0.5 ELSE 0.0 END AS score FROM votes
  UNION ALL
  SELECT rater, pos, right_arm AS arm, left_arm  AS opp,
         CASE result WHEN 'r' THEN 1.0 WHEN 't' THEN 0.5 ELSE 0.0 END AS score FROM votes;
CREATE VIEW pair_counts AS
  SELECT rater, arm, opp,
         count(*)                        AS games,
         sum(score = 1.0)                AS wins,
         sum(score = 0.5)                AS ties
  FROM outcomes GROUP BY rater, arm, opp;
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
