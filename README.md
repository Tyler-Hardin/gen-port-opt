### Intro

Uses genetic programming via OpenCL to compute semi-optimal portfolios under
incomplete restraints.

### Installation

Just get OpenCL working. Good luck.

And also numpy.

### Example output

```
Signal:
[-0.44093436  0.93071288 -0.61594826  0.41221145  0.76544189 -0.16351736
 -0.92200369 -0.82655901 -0.51413375  0.36817849 -0.75797117  0.64285469
 -0.69597197 -0.80592465  0.44992489 -0.31408849 -0.15248087 -0.82401484
  0.28724992  0.51718801 -0.58525956 -0.14409585 -0.65255517  0.01665593
 -0.79630935]
Prices:
[ 21.73966599  88.9185791   95.26016235  29.0491066   15.82285404
  46.15737152  93.57397461  16.3777771   45.9434967   98.39854431
  60.43335724  79.12461853  32.7732811   19.01377106  33.80745697
  86.79504395  43.95750046  39.22774887  39.7804985   45.2950325
  91.56587219  19.6781292   66.06242371  51.00658035  40.36725998]
ADV:
[ 6097109.5     4664963.5     2488903.      7669860.5     9566760.
  3557315.5     6755064.      1191086.25     626314.4375  3057164.
  6810295.5     8328944.5     9497237.      2324308.5     9405575.
  5999760.      1428979.125   8371371.5     6923162.5     9777449.
  7521718.      8718533.      8628937.      6620953.5     9115783.    ]
Spreads:
[ 0.24820328  1.26339722  1.30889893  0.40231895  0.0481596   0.11320496
  0.73011017  0.15657997  0.28573227  1.33651733  0.45953751  0.33732605
  0.22164917  0.09876823  0.46486664  0.88150024  0.34178162  0.59614563
  0.54877853  0.44728851  1.17875671  0.27637863  0.41187286  0.73875427
  0.54725266]
Portfolio:
[ 363  -40   61   -3 3899    7    0    3    1  189 -458 -155    3    0 -123
    7    3 -497   95  -27 -108   77 -310   -3  -11]
('Fitness:', 737341565.81613207)
('Max Participation:', 0.0064487149160181202)
('GMV:', 199972.98421955109)
('NMV:', 1113.6174364089966)
('Peason R bt alpha and port:', (0.37795099975819096, 0.062492563462954287))
```

Configuration is in source. All data was randomly generated to be
reasonably realistic.
