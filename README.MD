# CEIntel: AI engine of [CollabEd](https://github.com/wageeshar/collabed.git)

## Components

### Preprocessor
CEIntel Preprocessor is responsible for data cleansing, de-noising and formatting depending on the _source_ and _destination_ of data.
Source is where the data is ingested from (e.g., a database collection/table) and destination is the algorithm data is fed to.
Depending on the data load and/or structure, preprocessor dispatches jobs either to regular in-memory processor or distributed Apache Spark cluster.

### Intel provider


