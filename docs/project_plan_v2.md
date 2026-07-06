# План магистерской работы: сравнение GAN-архитектур для генерации синтетических данных сегментации дефектов стали

## 1. Исследовательский вопрос

Какая из GAN-архитектур (Pix2Pix, SPADE, StyleGAN2-ADA), обученных на генерацию изображений дефектов стали по маске, наиболее эффективна для создания синтетических данных, улучшающих качество модели сегментации дефектов — и коррелирует ли визуальное качество генерации (FID) с фактической пользой для downstream-задачи (mIoU)?

## 2. Датасет

**Severstal: Steel Defect Detection** (Kaggle) — реальные промышленные изображения листового проката с масками сегментации, 4 класса дефектов.

Препроцессинг: извлечение пар (изображение, маска) → train/val/test split, зафиксированный **test-split не участвует ни в обучении GAN, ни в обучении сегментатора** — используется только для финальной оценки mIoU, чтобы результаты были сопоставимы между всеми экспериментами.

## 3. Сравниваемые модели (генеративные)

| Модель | Уровень сложности | Ключевая идея |
|---|---|---|
| **Pix2Pix** | Базовый | Conditional GAN, U-Net генератор + PatchGAN дискриминатор, маска — входной канал |
| **SPADE** | Средний | Spatially-Adaptive Normalization — маска инжектируется на каждом слое генератора |
| **StyleGAN2-ADA** | Продвинутый | Style-based генератор + Adaptive Discriminator Augmentation, рассчитан на работу с ограниченными данными |

## 4. Фиксированная модель сегментации (измерительный прибор)

**U-Net (ResNet34 encoder)** — используется одинаково во всех экспериментах, чтобы изолировать эффект именно генеративной модели, а не вариативность самого сегментатора.

## 5. Матрица экспериментов

```
Baseline:        100% real / 0% synthetic            → mIoU (контрольная точка)
Pix2Pix:          75:25 | 50:50 | 25:75 (real:synth)  → mIoU × 3, FID
SPADE:            75:25 | 50:50 | 25:75                → mIoU × 3, FID
StyleGAN2-ADA:    75:25 | 50:50 | 25:75                → mIoU × 3, FID
```

Итого: 3 обученные GAN-модели + 10 прогонов сегментации + FID для каждой GAN.

## 6. Метрики

- **mIoU, Dice, pixel accuracy** — качество сегментации (основная метрика успеха)
- **FID (Fréchet Inception Distance)** — визуальное правдоподобие синтетики (независимо от downstream-задачи)
- **Качественный анализ** — визуальное сравнение сгенерированных дефектов, разбор типичных ошибок каждой GAN

## 7. Технологический стек

- **ОС:** Ubuntu 24.04, ROCm 7.2.4
- **Фреймворк:** PyTorch (ROCm build)
- **Сегментация:** `segmentation_models_pytorch` (U-Net, ResNet34)
- **GAN:** адаптированные референсные реализации Pix2Pix, SPADE, StyleGAN2-ADA (PyTorch)
- **Метрики:** `pytorch-fid` / `clean-fid`, кастомные mIoU/Dice
- **Трекинг экспериментов:** MLflow
- **Аугментации:** `albumentations`
- **Демо:** Streamlit — вкладка сегментации + вкладка "сгенерировать дефект" на лучшей GAN-модели

## 8. Timeline (14-16 недель)

| Недели | Этап |
|---|---|
| 1-2 | Препроцессинг данных (image-mask пары, сплиты), baseline-сегментация (100% real) |
| 3-5 | Pix2Pix: обучение + генерация синтетики под 3 соотношения |
| 6-9 | SPADE: обучение + генерация синтетики |
| 10-13 | StyleGAN2-ADA: обучение (самая тяжёлая часть) + генерация синтетики |
| 12-14 | Обучение U-Net на всех датасетах, сбор mIoU/FID (частично параллельно) |
| 14-15 | Сравнительный анализ, графики, качественный разбор ошибок, выводы |
| 15-16 | Демо-приложение, оформление текста диплома |

## 9. Структура репозитория

```
steel-defect-project/
├── data/
│   ├── raw/                      # исходный Severstal-архив
│   ├── processed/                # извлечённые пары image-mask
│   └── splits/                   # train/val/test индексы (фиксированные)
│
├── src/
│   ├── datasets/
│   │   ├── severstal_dataset.py  # Dataset/DataLoader для реальных данных
│   │   └── synthetic_dataset.py  # Dataset для смешивания real+synthetic по заданному ratio
│   │
│   ├── gans/
│   │   ├── pix2pix/
│   │   │   ├── model.py
│   │   │   ├── train.py
│   │   │   └── generate.py       # генерация синтетики по маскам test/train
│   │   ├── spade/
│   │   │   ├── model.py
│   │   │   ├── train.py
│   │   │   └── generate.py
│   │   └── stylegan2_ada/
│   │       ├── model.py
│   │       ├── train.py
│   │       └── generate.py
│   │
│   ├── segmentation/
│   │   ├── unet.py                # обёртка над segmentation_models_pytorch
│   │   ├── train.py                # единый скрипт обучения, принимает --ratio --gan-source
│   │   └── evaluate.py             # mIoU/Dice на фиксированном test-split
│   │
│   ├── metrics/
│   │   ├── fid.py
│   │   └── segmentation_metrics.py
│   │
│   └── demo/
│       └── app.py                  # Streamlit-приложение
│
├── configs/
│   ├── pix2pix.yaml
│   ├── spade.yaml
│   ├── stylegan2_ada.yaml
│   └── unet_segmentation.yaml
│
├── experiments/                    # результаты прогонов, mlruns/ (в .gitignore)
├── notebooks/                       # EDA, визуализация результатов
├── scripts/                          # bash-обёртки (запуск полного пайплайна)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
├── docs/
│   └── project_plan_v2.md           # этот файл
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 10. Git-ветки под новую схему

```
main
├── data-preprocessing
├── gan-pix2pix
├── gan-spade
├── gan-stylegan2-ada
├── segmentation-baseline
└── demo-app
```

Каждая GAN-ветка мёржится в `main` только после того, как из неё сгенерирован финальный синтетический датасет и подтверждены метрики FID.
