CREATE TABLE name2inchi
(
   inchi TEXT,
   inchikey TEXT,
   name TEXT
);

-- CREATE INDEX inchi_idx ON name2inchi (inchi); -- row size exceeds maximum
CREATE INDEX inchikey_idx ON name2inchi (inchikey);
CREATE INDEX name_idx ON name2inchi (name);
CREATE INDEX name_lower_idx on name2inchi (LOWER(name));
-- GRANT SELECT ON TABLE public.name2inchi TO readonly;  -- let the read only user use this table

create TABLE inchi2molid
(
    molid    int PRIMARY KEY,
    inchi    text UNIQUE NOT NULL,
    inchikey text UNIQUE NOT NULL,
    smiles text
);

-- create a small molecule record source
CREATE TABLE source(
   source_id INT PRIMARY KEY,
   source_name VARCHAR (50) UNIQUE NOT NULL
);

INSERT INTO source (source_id, source_name)
VALUES
(1, 'pubchem');

-- create function to do case insensitive name query

CREATE OR REPLACE FUNCTION name2inchi(names text[])
  RETURNS TABLE(name text, inchikey text, inchi text) AS
$func$

SELECT n.name, n.inchikey, n.inchi
FROM   name2inchi n
WHERE  lower(n.name) = ANY (lower($1::text)::text[]);

$func$ LANGUAGE sql;

-- create function to do inchi_key query

CREATE OR REPLACE FUNCTION inchi2name(inchi_key text)
  RETURNS TABLE(name text, inchikey text, inchi text) AS
$func$

SELECT n.name, n.inchikey, n.inchi
FROM   name2inchi n
WHERE  n.inchikey = $1;

$func$ LANGUAGE sql;