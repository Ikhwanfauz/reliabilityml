# FD001 data contract

Source: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

The NASA C-MAPSS dataset contains simulated engine trajectories. This project uses FD001 only. Downloaded files are kept in `data/raw/` and are excluded from Git.

## Raw files observed

| File | Rows | Columns | Meaning |
| --- | ---: | ---: | --- |
| `train_FD001.txt` | 20,631 | 26 | Run-to-failure histories for 100 training engines |
| `test_FD001.txt` | 13,096 | 26 | Truncated histories for 100 separate test engines |
| `RUL_FD001.txt` | 100 | 1 | Remaining cycles after the final observed cycle of each test engine |

The 26 history columns, in order, are:

1. Engine ID (`unit_id`)
2. Operating cycle (`cycle`)
3. Three operating settings (`setting_1` to `setting_3`)
4. Twenty-one sensor readings (`sensor_1` to `sensor_21`)

The text files have no header row and use whitespace between values.

## Target meaning

For a training engine, its last recorded cycle is its failure cycle. At an observed cycle, training RUL equals:

`last cycle of that training engine - current cycle`

The separate test RUL file gives one answer per test engine at the **last cycle available in its truncated history**. We will keep those official test answers out of model development and use them only for final evaluation after model selection.

RUL is measured in operating cycles, not calendar days or a failure probability.

## Raw file SHA-256 checksums

These hashes were calculated from the downloaded FD001 files using `certutil -hashfile <file> SHA256` on Windows.

| File | SHA-256 |
| --- | --- |
| `train_FD001.txt` | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
| `test_FD001.txt` | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |
| `RUL_FD001.txt` | `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca` |