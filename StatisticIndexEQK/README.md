# README
Data `summary_eqk_Rc[50]_ML[5].csv`
- The first column is `DateTime`, as denoted in the header.
- Columns from the 2nd to the last are variable `EQK` for each station; each header (column name) is the code of station.
- `EQK` is a series of `true`/`false`, indicating whether in the corresponding day an earthquake of magnitude >= `ML` occurs within the radius `Rc` of the station.
- The magnitude (`ML`) and radius (`Rc`) threshold is denoted on the file name.

Data `summary_stat.csv`
- The first column is `DateTime`, as denoted in the header.
- The column `stn` denotes the station.
- The column named `S` or `S_*` denotes the skewness, where the suffix (`_*`) denotes the channel of the instrument.
- The column named `K` or `K_*` denotes the kurtosis, where the suffix (`_*`) denotes the channel of the instrument.
- For geo-electric observations (e.g., `CHCH` and all others with station code of four characters), the variables are `S_NS`, `S_EW`, `K_NS`, `K_EW`.
- For geo-magnetic observations (e.g., `HL` and all others with station code of two characters), the variables are `S`, `S_x`, `S_y`, `S_z`, `K`, `K_x`, `K_y`, `K_z`.