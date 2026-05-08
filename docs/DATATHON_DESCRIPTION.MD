# Datathon Description

Target variable: `bilissel_performans_skoru`

Task type: supervised regression. The goal is to predict a bounded cognitive performance score for each row in `test_x.csv` and submit a CSV with the columns `id` and `bilissel_performans_skoru`.

## Files

- `data/raw/train.csv`: 56,000 labeled rows.
- `data/raw/test_x.csv`: 24,000 unlabeled rows.
- `data/raw/sample_submission.csv`: submission column template.

## Target

`bilissel_performans_skoru` is numeric and ranges from `0` to `10` in the training data. The modeling pipeline clips final predictions to this interval.

## Columns

| Column | Description |
| --- | --- |
| `id` | Row identifier. |
| `yas` | Age. |
| `cinsiyet` | Gender. |
| `meslek` | Occupation. |
| `vucut_kitle_indeksi` | Body mass index. |
| `ulke` | Country. Some country values appear in Turkish and English variants. |
| `rem_yuzdesi` | REM sleep percentage. |
| `derin_uyku_yuzdesi` | Deep sleep percentage. |
| `uykuya_dalma_suresi_dk` | Sleep latency in minutes. |
| `gecelik_uyanma_sayisi` | Number of nightly awakenings. |
| `uyku_oncesi_kafein_mg` | Caffeine intake before sleep in mg. |
| `uyku_oncesi_ekran_suresi_dk` | Screen time before sleep in minutes. |
| `gunluk_adim_sayisi` | Daily step count. |
| `sekerleme_suresi_dk` | Nap duration in minutes. |
| `stres_skoru` | Stress score. |
| `gunluk_calisma_saati` | Daily work/study hours. |
| `kronotip` | Chronotype. |
| `ruh_sagligi_durumu` | Mental health status. |
| `dinlenik_nabiz_bpm` | Resting heart rate. |
| `oda_sicakligi_celsius` | Room temperature in Celsius. |
| `hafta_sonu_uyku_farki_saat` | Weekend sleep schedule difference in hours. |
| `mevsim` | Season group. |
| `gun_tipi` | Weekday/weekend indicator. |

## Missing Values

Missing values appear in both train and test for:

- `meslek`
- `vucut_kitle_indeksi`
- `uyku_oncesi_kafein_mg`
- `stres_skoru`
- `kronotip`
- `ruh_sagligi_durumu`

The pipeline uses median imputation with missing indicators for numeric features and constant categorical imputation before one-hot encoding.
