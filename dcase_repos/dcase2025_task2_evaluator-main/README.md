# dcase2025\_task2\_evaluator
The **dcase2025\_task2\_evaluator** is a script for calculating the AUC, pAUC, precision, recall, and F1 scores from the anomaly score list for the [evaluation dataset](https://zenodo.org/records/15519362) in DCASE 2025 Challenge Task 2 "First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring."

[https://dcase.community/challenge2025/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring](https://dcase.community/challenge2025/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring)

## Description

The **dcase2025\_task2\_evaluator** consists of two scripts:

- `dcase2025_task2_evaluator.py`
    - This script outputs the AUC and pAUC scores by using:
      - Ground truth of the normal and anomaly labels
      - Anomaly scores for each wave file listed in the csv file for each machine type, section, and domain
      - Detection results for each wave file listed in the csv file for each machine type, section, and domain
- `03_evaluation_eval_data.sh`
    - This script execute `dcase2025_task2_evaluator.py`.

## Usage
### 1. Clone repository
Clone this repository from Github.

### 2. Prepare data
- Anomaly scores
    - Generate csv files `anomaly_score_<machine_type>_section_<section_index>_test.csv` and `decision_result_<machine_type>_section_<section_index>_test.csv` or `anomaly_score_DCASE2025T2<machine_type>_section_<section>_test_seed<seed><tag>_Eval.csv` and `decision_result_DCASE2025T2<machine_type>_section_<section>_test_seed<seed><tag>_Eval.csv` by using a system for the [evaluation dataset](https://zenodo.org/record/15519362). (The format information is described [here](https://dcase.community/challenge2025/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring#submission).)
- Rename the directory containing the csv files to a team name
- Move the directory into `./teams/`

### 3. Check directory structure
- ./dcase2025\_task2\_evaluator
    - /dcase2025\_task2\_evaluator.py
    - /03\_evaluation\_eval\_data.sh
    - /ground\_truth\_attributes
        - ground\_truth\_AutoTrash\_section\_00\_test.csv
        - ground\_truth\_BandSealer\_section\_00\_test.csv
        - ...
    - /ground\_truth\_data
        - ground\_truth\_AutoTrash\_section\_00\_test.csv
        - ground\_truth\_BandSealer\section\_00\_test.csv
        - ...
    - /ground\_truth\_domain
        - ground\_truth\_AutoTrash\_section\_00\_test.csv
        - ground\_truth\_BandSealer\_section\_00\_test.csv
        - ...
    - /teams
        - /\<team\_name\_1\>
            - /\<system\_name\_1\>
                - anomaly\_score\_AutoTrash\_section\_00\_test.csv
                - anomaly\_score\_BandSealer\_section\_00\_test.csv
                - ...
                - decision\_result\_ToyPet\_section\_00\_test.csv
                - decision\_result\_ToyRCCar\_section\_00\_test.csv
            - /\<system\_name\_2\>
                - anomaly\_score\_DCASE2025T2AutoTrash\_section\_00\_test\_seed\<--seed\>\<--tag\>\_Eval.csv
                - anomaly\_score\_DCASE2025T2BandSealer\_section\_00\_test\_seed\<--seed\>\<--tag\>\_Eval.csv
                - ...
                - decision\_result\_DCASE2025T2ToyPet\_section\_00\_test\_seed\<--seed\>\<--tag\>\_Eval.csv
                - decision\_result\_DCASE2025T2ToyRCCar\_section\_00\_test\_seed\<--seed\>\<--tag\>\_Eval.csv
        - /\<team\_name\_2\>
            - /\<system\_name\_3\>
                - anomaly\_score\_AutoTrash\_section\_00\_test.csv
                - anomaly\_score\_BandSealer\_section\_00\test.csv
                - ...
                - decision\_result\_ToyPet\_section\_00\_test.csv
                - decision\_result\_ToyRCCar\_section\_00\_test.csv
        - ...
    - /teams\_result
        - \<system\_name\_1\>\_result.csv
        - \<system\_name\_2\>\_result.csv
        - \<system\_name\_3\>_result.csv
        - ...
    - /teams\_additional\_result \*`out_all==True`
        - teams\_official\_score.csv
        - teams\_official\_score\_paper.csv
        - teams\_section\_00\_auc.csv
        - teams\_section\_00\_score.csv
        - /\<system\_name\_1\>
            - official\_score.csv
            - \<system\_name\_1\>\_AutoTrash\_section\_00\_anm\_score.png
            - ...
            - \<system\_name\_1\>\_ToyRCCar\_section\_00\_anm\_score.png
        - /\<system\_name\_2\>
            - official\_score.csv
            - \<system\_name\_2\>\_AutoTrash\_section\_00\_anm\_score.png
            - ...
            - \<system\_name\_2\>\_ToyRCCar\_section\_00\_anm\_score.png
        - /\<system\_name\_3\>
            - official\_score.csv
            - \<system\_name\_3\>\_AutoTrash\_section\_00\_anm\_score.png
            - ...
            - \<system\_name\_3\>\_ToyRCCar\_section\_00\_anm\_score.png
        - ...
    - /tools
        - plot\_anm\_score.py
        - test\_plots.py
    - /README.md


### 4. Change parameters
The parameters are defined in the script `dcase2025_task2_evaluator.py` as follows.
- **MAX\_FPR**
    - The FPR threshold for pAUC : default 0.1
- **--result\_dir**
    - The output directory : default `./teams_result/`
- **--teams\_root\_dir**
    - Directory containing team results. : default `./teams/`
- **--dir\_depth**
    - What depth to search `--teams_root_dir` using glob. : default `2`
    - If --dir\_depth=2, then `glob.glob(<teams_root_dir>/*/*)`
- **--tag**
    - File name tag. : default `_id(0_)`
    - If using filename is DCASE2025 baseline style, change parameters as necessary. 
- **--seed**
    - Seed used during train. : default `13711`
    - If using filename is DCASE2025 baseline style, change parameters as necessary.
- **--out\_all**
    - If this parameter is `True`, export supplemental data. : default `False`
- **--additional\_result\_dir**
    - The output additional results directory. : default `./teams_additional_result/`
    - Used when `--out_all==True`.

### 5. Run script
Run the script `dcase2025_task2_evaluator.py`
```
$ python dcase2025_task2_evaluator.py
```
or
```
$ bash 03_evaluation_eval_data.sh
```
The script `dcase2025_task2_evaluator.py` calculates the AUC, pAUC, precision, recall, and F1 scores for each machine type, section, and domain and output the calculated scores into the csv files (`<system_name_1>_result.csv`, `<system_name_2>_result.csv`, ...) in **--result\_dir** (default: `./teams_result/`).
If **--out\_all=True**, each team results are then aggregated into a csv file (`teams_official_score.csv`, `teams_official_score_paper.csv`) in **--additional\_result\_dir** (default: `./teams_additional_result`).

### 6. Check results
You can check the AUC, pAUC, precision, recall, and F1 scores in the `<system_name_N>_result.csv` in **--result\_dir**.
The AUC, pAUC, precision, recall, and F1 scores for each machine type, section, and domain are listed as follows:

`<section_name_N>_result.csv`
```
AutoTrash
section,AUC (all),AUC (source),AUC (target),pAUC,precision (source),precision (target),recall (source),recall (target),F1 score (source),F1 score (target)
00,0.5769000000000001,0.8102,0.3436,0.5421052631578948,0.5119047619047619,0.5,0.86,1.0,0.6417910447761195,0.6666666666666666
,,AUC,pAUC,precision,recall,F1 score
arithmetic mean,,0.5769,0.5421052631578948,0.5059523809523809,0.9299999999999999,0.6542288557213931
harmonic mean,,0.48255281677933787,0.5421052631578948,0.5058823529411764,0.9247311827956988,0.6539923954372623
source harmonic mean,,0.8102,0.5421052631578948,0.5119047619047619,0.86,0.6417910447761195
target harmonic mean,,0.3436,0.5421052631578948,0.5,1.0,0.6666666666666666

...

ToyRCCar
section,AUC (all),AUC (source),AUC (target),pAUC,precision (source),precision (target),recall (source),recall (target),F1 score (source),F1 score (target)
00,0.5777999999999999,0.5284,0.6271999999999999,0.5552631578947368,0.6818181818181818,0.4666666666666667,0.6,0.14,0.6382978723404256,0.2153846153846154
,,AUC,pAUC,precision,recall,F1 score
arithmetic mean,,0.5777999999999999,0.5552631578947368,0.5742424242424242,0.37,0.4268412438625205
harmonic mean,,0.5735764624437522,0.5552631578947368,0.554089709762533,0.22702702702702707,0.3220858895705522
source harmonic mean,,0.5284,0.5552631578947368,0.6818181818181818,0.6,0.6382978723404256
target harmonic mean,,0.6271999999999999,0.5552631578947368,0.4666666666666667,0.14,0.2153846153846154

...

,,AUC,pAUC,precision,recall,F1 score
"arithmetic mean over all machine types, sections, and domains",,0.5858625,0.5468421052631579,0.5183191989199928,0.81,0.6104748915566067
"harmonic mean over all machine types, sections, and domains",,0.5437772342298658,0.5452967030441773,0.5150751507085616,0.6207167119350003,0.5629829979642624
"source harmonic mean over all machine types, sections, and domains",,0.6879822239700398,0.5452967030441773,0.5281415194743965,0.6953393434776113,0.6003160139808418
"target harmonic mean over all machine types, sections, and domains",,0.44954916968961445,0.5452967030441773,0.5026397039837188,0.5605585275183607,0.5300215218366398

official score,,0.5442827820713174
official score ci95,,1.271407576916618e-05
```

Aggregated results for each baseline are listed as follows:

```_seed13711_official_score_paper.csv
System,metric,h-mean,a-mean,AutoTrash,HomeCamera,ToyPet,ToyRCCar,BandSealer,CoffeeGrinder,Polisher,ScrewFeeder
DCASE2025_baseline_task2_MAHALA,AUC (source),0.719933864244911,0.729725,0.7726000000000001,0.8616,0.6981999999999999,0.5586,0.7638,0.7498,0.7041999999999999,0.729
DCASE2025_baseline_task2_MAHALA,AUC (target),0.4788331261490967,0.508175,0.526,0.42640000000000006,0.509,0.5548,0.3268,0.4042,0.5278,0.7904000000000001
DCASE2025_baseline_task2_MAHALA,"pAUC (source, target)",0.5459161077739156,0.5515131578947368,0.541578947368421,0.5184210526315789,0.5684210526315789,0.54,0.49105263157894735,0.5142105263157895,0.5378947368421052,0.7005263157894737
DCASE2025_baseline_task2_MAHALA,TOTAL score,0.5650558189601554,0.596471052631579,,,,,,,,
DCASE2025_baseline_task2_MSE,AUC (source),0.6879822239700398,0.6996249999999999,0.8102,0.8140000000000001,0.677,0.5284,0.7198,0.7303999999999999,0.6686000000000001,0.6486
DCASE2025_baseline_task2_MSE,AUC (target),0.44954916968961445,0.4721,0.3436,0.4976,0.36699999999999994,0.6271999999999999,0.3956,0.4436,0.443,0.6592
DCASE2025_baseline_task2_MSE,"pAUC (source, target)",0.5452967030441773,0.5468421052631579,0.5421052631578948,0.5284210526315789,0.55,0.5552631578947368,0.5205263157894737,0.5342105263157895,0.5231578947368422,0.6210526315789474
DCASE2025_baseline_task2_MSE,TOTAL score,0.5442827820713174,0.5728557017543859,,,,,,,,

```


## Citation

If you use this system, please cite all the following four papers:

+ Tomoya Nishida, Noboru Harada, Daisuke Niizumi, Davide Albertini, Roberto Sannino, Simone Pradolini, Filippo Augusti, Keisuke Imoto, Kota Dohi, Harsh Purohit, Takashi Endo, and Yohei Kawaguchi. Description and discussion on DCASE 2025 challenge task 2: first-shot unsupervised anomalous sound detection for machine condition monitoring. In arXiv e-prints: 2506.10097, 2025. [URL](https://arxiv.org/pdf/2506.10097.pdf)
+ Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, and Shoichiro Saito. ToyADMOS2: another dataset of miniature-machine operating sounds for anomalous sound detection under domain shift conditions. In Proceedings of the Detection and Classification of Acoustic Scenes and Events Workshop (DCASE), 1–5. Barcelona, Spain, November 2021. [URL](https://dcase.community/documents/workshop2021/proceedings/DCASE2021Workshop_Harada_6.pdf)
+ Kota Dohi, Tomoya Nishida, Harsh Purohit, Ryo Tanabe, Takashi Endo, Masaaki Yamamoto, Yuki Nikaido, and Yohei Kawaguchi. MIMII DG: sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task. In Proceedings of the 7th Detection and Classification of Acoustic Scenes and Events 2022 Workshop (DCASE2022). Nancy, France, November 2022. [URL](https://dcase.community/documents/workshop2022/proceedings/DCASE2022Workshop_Dohi_62.pdf)
+ Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, and Masahiro Yasuda. First-shot anomaly detection for machine condition monitoring: a domain generalization baseline. Proceedings of 31st European Signal Processing Conference (EUSIPCO), pages 191–195, 2023. [URL](https://eurasip.org/Proceedings/Eusipco/Eusipco2023/pdfs/0000191.pdf)