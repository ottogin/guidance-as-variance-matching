-- Owner settings, e.g. which people are left out of the leaderboard: key 'excluded' -> JSON list of person keys.
CREATE TABLE settings (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
