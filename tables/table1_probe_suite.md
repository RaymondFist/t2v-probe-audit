| probe | operation | pixel_values | frame_order | adjacent_pairs | role | reported |
|---|---|---|---|---|---|---|
| I1 reverse | reverse the whole clip | unchanged | reversed | same multiset | anchor diagnostic | yes |
| I2 window reverse | reverse inside windows of 4 | unchanged | partly reversed | mostly same | anchor diagnostic | no |
| I3 shuffle | seeded permutation | unchanged | destroyed | changed | order sensitivity | yes |
| I4 2x | drop every other frame | subset | kept | subset | rate, faster | no |
| I5 0.5x | duplicate every frame | unchanged | kept | duplicated | rate, slower | yes |
| I6 block jitter | shift highest-variance cell | locally changed | kept | changed | local inconsistency | no |
