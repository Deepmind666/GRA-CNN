# Enhanced Statistical Report (v2)

- comparisons_valid: 46
- wins_gra: 17
- losses_gra: 1
- ties: 28
- insufficient: 14

## Normality checks (Shapiro-Wilk on paired differences)

- resnet110/cifar100/r0.7/CHIP: W=0.8996, p=0.4079 [PASS]
- resnet110/cifar100/r0.7/L1: W=0.8112, p=0.0997 [PASS]
- resnet110/cifar100/r0.7/Taylor: W=0.9687, p=0.8672 [PASS]
- resnet110/cifar100/r0.9/CHIP: W=0.8663, p=0.2517 [PASS]
- resnet110/cifar100/r0.9/L1: W=0.8608, p=0.2313 [PASS]
- resnet110/cifar100/r0.9/Taylor: W=0.8767, p=0.2948 [PASS]
- resnet18/tinyimagenet/r0.7/CHIP: W=0.9694, p=0.6642 [PASS]
- resnet18/tinyimagenet/r0.7/FPGM: W=1.0, p=1.0 [PASS]
- resnet18/tinyimagenet/r0.7/HRank: W=0.794, p=0.1001 [PASS]
- resnet18/tinyimagenet/r0.7/L1: W=0.9423, p=0.5367 [PASS]
- resnet18/tinyimagenet/r0.9/CHIP: W=0.9735, p=0.6878 [PASS]
- resnet18/tinyimagenet/r0.9/L1: W=0.9018, p=0.3914 [PASS]
- resnet56/cifar10/r0.7/CHIP: W=0.8829, p=0.3226 [PASS]
- resnet56/cifar10/r0.7/Taylor: W=0.9756, p=0.9097 [PASS]
- resnet56/cifar10/r0.9/CHIP: W=0.7956, p=0.0745 [PASS]
- resnet56/cifar10/r0.9/Taylor: W=0.9355, p=0.634 [PASS]
- resnet56/cifar100/r0.6/CHIP: W=0.8685, p=0.2604 [PASS]
- resnet56/cifar100/r0.6/FPGM: W=0.7516, p=0.0308 [FAIL]
- resnet56/cifar100/r0.6/HRank: W=0.7623, p=0.0386 [FAIL]
- resnet56/cifar100/r0.6/L1: W=0.8349, p=0.1513 [PASS]
- resnet56/cifar100/r0.6/Taylor: W=0.8645, p=0.2447 [PASS]
- resnet56/cifar100/r0.7/CHIP: W=0.8906, p=0.3601 [PASS]
- resnet56/cifar100/r0.7/FPGM: W=0.9397, p=0.6641 [PASS]
- resnet56/cifar100/r0.7/HRank: W=0.9003, p=0.4115 [PASS]
- resnet56/cifar100/r0.7/L1: W=0.8984, p=0.4013 [PASS]
- resnet56/cifar100/r0.7/Taylor: W=0.8141, p=0.1051 [PASS]
- resnet56/cifar100/r0.8/CHIP: W=0.9398, p=0.6642 [PASS]
- resnet56/cifar100/r0.8/FPGM: W=0.9342, p=0.6252 [PASS]
- resnet56/cifar100/r0.8/HRank: W=0.7884, p=0.065 [PASS]
- resnet56/cifar100/r0.8/L1: W=0.8781, p=0.3009 [PASS]
- resnet56/cifar100/r0.8/Taylor: W=0.9807, p=0.9385 [PASS]
- resnet56/cifar100/r0.9/CHIP: W=0.9657, p=0.847 [PASS]
- resnet56/cifar100/r0.9/FPGM: W=0.8263, p=0.1304 [PASS]
- resnet56/cifar100/r0.9/HRank: W=0.9739, p=0.8994 [PASS]
- resnet56/cifar100/r0.9/L1: W=0.958, p=0.7937 [PASS]
- resnet56/cifar100/r0.9/Taylor: W=0.9728, p=0.8928 [PASS]
- vgg16/cifar100/r0.7/CHIP: W=0.9302, p=0.5977 [PASS]
- vgg16/cifar100/r0.7/FPGM: W=0.9629, p=0.8283 [PASS]
- vgg16/cifar100/r0.7/HRank: W=0.922, p=0.5428 [PASS]
- vgg16/cifar100/r0.7/L1: W=0.8211, p=0.1191 [PASS]
- vgg16/cifar100/r0.7/Taylor: W=0.9099, p=0.4672 [PASS]
- vgg16/cifar100/r0.9/CHIP: W=0.9767, p=0.9161 [PASS]
- vgg16/cifar100/r0.9/FPGM: W=0.875, p=0.2873 [PASS]
- vgg16/cifar100/r0.9/HRank: W=0.9934, p=0.9901 [PASS]
- vgg16/cifar100/r0.9/L1: W=0.7223, p=0.0162 [FAIL]
- vgg16/cifar100/r0.9/Taylor: W=0.9388, p=0.6572 [PASS]

## Full results table

| arch | ratio | baseline | n | diff | t-p | holm-p | perm-p | Hedges g | 95% CI | winner |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| resnet110 | 0.7 | CHIP | 5 | -0.616 | 0.006777 | 0.176209 | 0.060994 | -1.84 | [-0.802, -0.398] | tie |
| resnet110 | 0.7 | FPGM | 0 | - | - | - | - | - | - | insufficient |
| resnet110 | 0.7 | HRank | 0 | - | - | - | - | - | - | insufficient |
| resnet110 | 0.7 | L1 | 5 | -0.358 | 0.00857 | 0.211283 | 0.060994 | -1.7217 | [-0.466, -0.21] | tie |
| resnet110 | 0.7 | Taylor | 5 | 0.33 | 0.022227 | 0.426723 | 0.060994 | 1.2975 | [0.184, 0.504] | tie |
| resnet110 | 0.9 | CHIP | 5 | 0.138 | 0.141422 | 1.0 | 0.248875 | 0.6543 | [0.0138, 0.272] | tie |
| resnet110 | 0.9 | FPGM | 0 | - | - | - | - | - | - | insufficient |
| resnet110 | 0.9 | HRank | 0 | - | - | - | - | - | - | insufficient |
| resnet110 | 0.9 | L1 | 5 | 0.21 | 0.175347 | 1.0 | 0.245975 | 0.5885 | [-0.008, 0.434] | tie |
| resnet110 | 0.9 | Taylor | 5 | 2.178 | 8e-06 | 0.000334 | 0.060994 | 10.5902 | [2.04, 2.286] | GRA |
| resnet18 | 0.7 | CHIP | 3 | 0.3867 | 0.036739 | 0.587822 | 0.252775 | 1.6734 | [0.27, 0.53] | tie |
| resnet18 | 0.7 | FPGM | 3 | 0.25 | 0.040588 | 0.587822 | 0.252775 | 1.5873 | [0.16, 0.34] | tie |
| resnet18 | 0.7 | HRank | 3 | 0.33 | 0.095692 | 0.956924 | 0.252775 | 0.9884 | [0.21, 0.55] | tie |
| resnet18 | 0.7 | L1 | 3 | 0.45 | 0.008451 | 0.211283 | 0.252775 | 3.5659 | [0.39, 0.53] | tie |
| resnet18 | 0.7 | Taylor | 0 | - | - | - | - | - | - | insufficient |
| resnet18 | 0.9 | CHIP | 3 | -0.3733 | 0.150344 | 1.0 | 0.252775 | -0.7517 | [-0.68, -0.12] | tie |
| resnet18 | 0.9 | FPGM | 0 | - | - | - | - | - | - | insufficient |
| resnet18 | 0.9 | HRank | 0 | - | - | - | - | - | - | insufficient |
| resnet18 | 0.9 | L1 | 3 | 0.2867 | 0.021336 | 0.426723 | 0.252775 | 2.2223 | [0.23, 0.37] | tie |
| resnet18 | 0.9 | Taylor | 0 | - | - | - | - | - | - | insufficient |
| resnet56 | 0.7 | CHIP | 5 | -0.168 | 0.027847 | 0.473398 | 0.060994 | -1.2084 | [-0.262, -0.086] | tie |
| resnet56 | 0.7 | FPGM | 0 | - | - | - | - | - | - | insufficient |
| resnet56 | 0.7 | HRank | 0 | - | - | - | - | - | - | insufficient |
| resnet56 | 0.7 | L1 | 0 | - | - | - | - | - | - | insufficient |
| resnet56 | 0.7 | Taylor | 5 | 0.652 | 0.000188 | 0.006952 | 0.060994 | 4.7379 | [0.57, 0.74] | GRA |
| resnet56 | 0.9 | CHIP | 5 | -0.268 | 0.197143 | 1.0 | 0.251575 | -0.5529 | [-0.508, 0.064] | tie |
| resnet56 | 0.9 | FPGM | 0 | - | - | - | - | - | - | insufficient |
| resnet56 | 0.9 | HRank | 0 | - | - | - | - | - | - | insufficient |
| resnet56 | 0.9 | L1 | 0 | - | - | - | - | - | - | insufficient |
| resnet56 | 0.9 | Taylor | 5 | 1.074 | 2.1e-05 | 0.000865 | 0.060994 | 8.2372 | [0.998, 1.158] | GRA |
| resnet56 | 0.6 | CHIP | 5 | 0.108 | 0.323941 | 1.0 | 0.312469 | 0.4021 | [-0.07, 0.268] | tie |
| resnet56 | 0.6 | FPGM | 5 | -0.552 | 0.000665 | 0.021943 | 0.060994 | -3.4254 | [-0.67, -0.472] | FPGM |
| resnet56 | 0.6 | HRank | 5 | 0.724 | 2.2e-05 | 0.00089 | 0.060994 | 8.1264 | [0.66, 0.768] | GRA |
| resnet56 | 0.6 | L1 | 5 | -0.256 | 0.018198 | 0.38216 | 0.060994 | -1.3798 | [-0.358, -0.126] | tie |
| resnet56 | 0.6 | Taylor | 5 | 0.11 | 0.085353 | 0.938879 | 0.245975 | 0.8136 | [0.024, 0.196] | tie |
| resnet56 | 0.7 | CHIP | 5 | 0.512 | 0.000614 | 0.020874 | 0.060994 | 3.4969 | [0.42, 0.604] | GRA |
| resnet56 | 0.7 | FPGM | 5 | 2.034 | 0.0 | 9e-06 | 0.060994 | 26.6432 | [1.992, 2.082] | GRA |
| resnet56 | 0.7 | HRank | 5 | 3.302 | 7e-06 | 0.000294 | 0.060994 | 10.9905 | [3.138, 3.5] | GRA |
| resnet56 | 0.7 | L1 | 5 | -0.174 | 0.180243 | 1.0 | 0.248875 | -0.5801 | [-0.376, -0.012] | tie |
| resnet56 | 0.7 | Taylor | 5 | 0.476 | 0.040508 | 0.587822 | 0.060994 | 1.0682 | [0.202, 0.746] | tie |
| resnet56 | 0.8 | CHIP | 5 | 0.494 | 0.004287 | 0.11574 | 0.060994 | 2.0893 | [0.34, 0.646] | tie |
| resnet56 | 0.8 | FPGM | 5 | 0.374 | 0.015897 | 0.357256 | 0.060994 | 1.4374 | [0.214, 0.548] | tie |
| resnet56 | 0.8 | HRank | 5 | 1.102 | 9.8e-05 | 0.003737 | 0.060994 | 5.5847 | [0.962, 1.198] | GRA |
| resnet56 | 0.8 | L1 | 5 | -0.342 | 0.037176 | 0.587822 | 0.124788 | -1.0995 | [-0.518, -0.134] | tie |
| resnet56 | 0.8 | Taylor | 5 | 1.164 | 0.000748 | 0.023541 | 0.060994 | 3.3219 | [0.94, 1.382] | GRA |
| resnet56 | 0.9 | CHIP | 5 | 0.538 | 0.002006 | 0.056164 | 0.060994 | 2.5644 | [0.404, 0.66] | tie |
| resnet56 | 0.9 | FPGM | 5 | 7.152 | 4e-06 | 0.000183 | 0.060994 | 12.4583 | [6.756, 7.444] | GRA |
| resnet56 | 0.9 | HRank | 5 | 10.082 | 1.5e-05 | 0.000625 | 0.060994 | 8.9931 | [9.414, 10.848] | GRA |
| resnet56 | 0.9 | L1 | 5 | 0.036 | 0.745206 | 1.0 | 0.877812 | 0.1246 | [-0.146, 0.208] | tie |
| resnet56 | 0.9 | Taylor | 5 | 1.502 | 0.000951 | 0.028533 | 0.060994 | 3.1211 | [1.184, 1.802] | GRA |
| vgg16 | 0.7 | CHIP | 5 | -0.058 | 0.708746 | 1.0 | 0.747125 | -0.1436 | [-0.28, 0.206] | tie |
| vgg16 | 0.7 | FPGM | 5 | 6.686 | 0.000407 | 0.014658 | 0.060994 | 3.8875 | [5.564, 7.696] | GRA |
| vgg16 | 0.7 | HRank | 5 | 4.812 | 0.000736 | 0.023541 | 0.060994 | 3.3368 | [3.99, 5.838] | GRA |
| vgg16 | 0.7 | L1 | 5 | 1.558 | 0.025052 | 0.450935 | 0.060994 | 1.2497 | [0.888, 2.434] | tie |
| vgg16 | 0.7 | Taylor | 5 | 2.862 | 0.001407 | 0.040792 | 0.060994 | 2.8169 | [2.268, 3.546] | GRA |
| vgg16 | 0.9 | CHIP | 5 | 0.148 | 0.769779 | 1.0 | 0.811519 | 0.1121 | [-0.634, 0.946] | tie |
| vgg16 | 0.9 | FPGM | 5 | 8.324 | 0.000525 | 0.018382 | 0.060994 | 3.6408 | [6.874, 9.774] | GRA |
| vgg16 | 0.9 | HRank | 5 | 7.324 | 8.6e-05 | 0.003365 | 0.060994 | 5.7729 | [6.508, 8.11] | GRA |
| vgg16 | 0.9 | L1 | 5 | 25.586 | 0.061597 | 0.739165 | 0.123088 | 0.9216 | [8.712, 41.9] | tie |
| vgg16 | 0.9 | Taylor | 5 | 0.564 | 0.015533 | 0.357256 | 0.060994 | 1.4474 | [0.33, 0.798] | tie |