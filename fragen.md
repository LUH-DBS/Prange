<!-- LTex: enabled=false -->

# Fragen und Themen zur Bachelorarbeit

## Ergebnisse

Ergebnisse nach zwei Minuten Training mit Spalten aus 900 Tabellen (100-999 aus gittables).
Die nicht trivialen Spalten (keine doppelten Werte in den ersten x Zeilen) stammen aus 500 Tabellen (1000-1499 aus gittables).

- 5 Spalten
  - Accuracy score: 0.9703825586178527
  - Non trivial accuracy score: 0.9196687370600414
- 10 Spalten
  - Accuracy score: 0.9831345125462773
  - Non trivial accuracy score: 0.9569767441860465
- 20 Spalten
  - Accuracy score: 0.9987659399424106
  - Non trivial accuracy score: 0.9955357142857143
- 50 Spalten
  - Accuracy score: 0.9995886466474702
  - Non trivial accuracy score: 1.0

| Spalten | Accuracy | n.t. Accuracy |
| ------- | -------- | ------------- |
| 5       | 97.04%   | 91.97%        |
| 10      | 98.31%   | 95.70%        |
| 20      | 99.88%   | 99.55%        |
| 50      | 99.96%   | 100%          |

## Fragen

## Ideen

- nicht nur accuracy messen, sondern auch precision (wie viele wurden korrekt als Kandidaten erkannt)
- csv der Tabellen aus dem Internet (<https://gittables.github.io/>)

## Anmerkungen

- Begriffsdefinitionen nicht vergessen (als Beispiel Sec. 2 in [Key Discovery])

[data profiling]: https://link.springer.com/article/10.1007/s00778-015-0389-y "Profiling relational data: a survey"
[ducc]: https://www.vldb.org/pvldb/vol7/p301-heise.pdf
[metronome projekt]: https://hpi.de/naumann/projects/data-profiling-and-analytics/metanome-data-profiling.html
[dataxformer]: https://cs.uwaterloo.ca/~ilyas/papers/AbedjanICDE16.pdf
