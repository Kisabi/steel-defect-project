# steel-defect-project

Магистерская работа: сравнение GAN-архитектур (Pix2Pix, SPADE, StyleGAN2-ADA)
для генерации синтетических данных, улучшающих качество сегментации дефектов стали
(Severstal dataset).

См. docs/project_plan_v2.md для полного плана и обоснования.

## Быстрый старт

    docker compose -f docker/docker-compose.yml build
    docker compose -f docker/docker-compose.yml up -d
    docker compose -f docker/docker-compose.yml exec ml-dev bash

## Структура

- data/        — датасеты (raw / processed / splits), не в git
- src/         — весь код (datasets, gans, segmentation, metrics, demo)
- configs/     — YAML-конфиги под каждую модель
- experiments/ — результаты прогонов, MLflow-логи
- docs/        — план проекта, документация для диплома
