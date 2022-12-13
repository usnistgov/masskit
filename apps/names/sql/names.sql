-- temporary table used to load in names to name2inchi
create table t
(
    index int,
    cid int,
    name text,
    inchi text,
    inchikey text
);

-- this only is allowed for superusers
-- copy t (index, cid, name, inchi, inchikey)
-- from 'C:\Users\lyg\data\chem\pubchem\name2inchi.csv'
-- with (format csv);

-- instead, use from psql
\copy t from program 'cmd /c "type C:\Users\lyg\data\chem\pubchem\name2inchi.csv"' WITH DELIMITER ',' CSV QUOTE AS '"' HEADER;
-- the odd syntax with cmd is necessary to load a multi GB file.


-- need to drop indexes to speed load into main tables
drop INDEX inchikey_idx;
drop INDEX name_idx;

insert into name2inchi (inchi, inchikey, name)
select inchi, inchikey, name
from public.t;

--recreate indexes
CREATE INDEX inchikey_idx ON name2inchi (inchikey);
CREATE INDEX name_idx ON name2inchi (name);
CREATE INDEX name_lower_idx on name2inchi (LOWER(name));

drop table t

