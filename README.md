# Qolda pilot study — казахоязычный разрыв в VLM-навигации мобильных роботов


## Структура

```
main.py                          CLI: python main.py <check> [флаги]
qolda_pilot/
  client.py                      QoldaClient (OpenAI-совместимый, lmdeploy) + MockQoldaClient
  config.py                      QoldaConfig: base_url, model, temperature, ...
  schemas.py                     Pydantic: ParsedCommand, Hypothesis, VerificationResult
  prompts.py                     Все казахскоязычные промпты, по одному месту на узел конвейера
  json_utils.py                  Устойчивое извлечение JSON из ответа модели (think-блоки, markdown-ограждения, обрезанный JSON)
  metrics.py                     Latency stats, Wilson CI, Brier score, ECE, Cohen's d
  io_utils.py                    Запись JSONL/CSV
  util.py                        ensure_utf8_stdout() — без него Windows-консоль падает на казахской кириллице
data/
  commands_kk.json                30 казахских команд навигации (15 культурных объектов, 15 универсальных) с золотым target_kk/en/rooms
  cultural_objects.json           34 казахских слова (18 культурных, 16 универсальных) с английским глоссом — для проверки #5
  crops_manifest.example.csv      Формат манифеста для проверки #3; нужно заменить реальными фотографиями
scripts/
  00_launch_server.sh/.ps1        Команды запуска Qolda через lmdeploy (OpenAI-совместимый сервер)
  01_bench_latency.py             Проверка #1: латентность Think vs No-Think
  02_eval_command_parsing.py      Проверка #2: надёжность JSON при разборе команды
  03_eval_candidate_verification.py  Проверка #3: калибровка да/нет-проверки кандидата (logprobs или self-consistency)
  04_eval_grounding.py            Проверка #4: умеет ли модель выдавать bounding box
  05_cultural_gap_study.py        Проверка #5: разрыв kk/en на культурной лексике (NLLB vs Qolda vs SigLIP2)
report/
  make_tables.py                  Собирает results/*.csv в report/pilot_report.md
results/                          Выходные CSV/JSONL (создаётся автоматически)
```

## Установка

```bash
pip install -r requirements.txt
```

`torch`/`transformers` нужны только для проверки #5 (NLLB, SigLIP2) — если
хотите прогнать только 1–4, можно их не ставить и просто не запускать `gap`.

## Запуск сервера Qolda

Реальный репозиторий модели — [`issai/Qolda`](https://huggingface.co/issai/Qolda)
(Apache 2.0, 4.3B, InternViT-300M + Qwen3-4B). lmdeploy на Windows нативно
поддерживается ограниченно; надёжный путь — WSL2.

```bash
# внутри WSL2
pip install "lmdeploy>=0.9.1"
bash scripts/00_launch_server.sh issai/Qolda qolda-4b 23333
```

`--backend pytorch` внутри скрипта — не опция, а обязательное требование:
архитектура Qolda (InternViT + Qwen3) не поддерживается турбомайнд-бэкендом
lmdeploy. Официально квантованных версий (AWQ/4-bit) для Qolda пока нет, так
что сервер поднимает bf16-веса (~8.6GB) — скрипт уже подрезает
`--session-len`/`--cache-max-entry-count` под карту с 12GB VRAM.

Подробности и 4-битный режим — в самом файле. **Важно**: значение, которое вы
передаёте в `--model-name`, — это то, что нужно указать в `--model` / `QOLDA_MODEL`
во всех остальных скриптах (это ID в served API, не HF repo id).

Проверка, что сервер жив:
```bash
curl http://localhost:23333/v1/models
```

Think/No-Think — это параметр **каждого запроса** (`enable_thinking` в
`extra_body`), а не сервера, поэтому пересоздавать сервер для сравнения
режимов не нужно (см. `qolda_pilot/client.py`).

## Запуск проверок

Единая точка входа — `main.py` (короткие имена) либо сами скрипты напрямую
(полные флаги через `--help`).

```bash
# быстрый dry-run всего конвейера без GPU
python main.py all --mock

# реальный прогон, по одной проверке за раз
python main.py latency   --base-url http://localhost:23333/v1 --model qolda-4b --n-trials 20
python main.py parsing   --base-url http://localhost:23333/v1 --model qolda-4b --modes no_think,think
python main.py verify    --base-url http://localhost:23333/v1 --model qolda-4b --manifest data/crops_manifest.csv
python main.py grounding --base-url http://localhost:23333/v1 --model qolda-4b --images data/grounding_probe.csv
python main.py gap       --base-url http://localhost:23333/v1 --model qolda-4b

python main.py report   # собрать все results/*.csv в report/pilot_report.md
```

`QOLDA_BASE_URL` / `QOLDA_MODEL` можно задать переменными окружения вместо
флагов — см. `qolda_pilot/config.py`.

### Проверка #3 требует реальных фото

`data/crops_manifest.example.csv` — только образец формата (`image_path,
object_kk, object_en, gold_label`), пути в нём не существуют. По плану
пилота: 30–50 фото на объект, из них выберите положительные и «сложные
отрицательные» (похожий, но другой предмет — например шкаф вместо сандық)
кропы, соберите свой `manifest.csv` и передайте `--manifest`.

### Проверка #4: формат bounding box неизвестен заранее

Скрипт пробует сразу несколько конвенций (InternVL `<box>[[...]]</box>`,
Qwen-VL `<|box_start|>`, JSON `{"bbox": [...]}`, голый список чисел) и
сообщает, какая сработала. Если ни одна не сработала — откройте
`results/04_grounding_raw.jsonl` и посмотрите на сырые ответы руками,
прежде чем делать вывод, что модель не умеет grounding.

### Проверка #5: три независимых сигнала разрыва

1. **NLLB-200** (`facebook/nllb-200-distilled-600M`) переводит казахское
   слово на английский; сравнение с золотым глоссом — лексическое
   перекрытие (Jaccard).
2. **Qolda** сама даёт английский глосс тем же способом — показывает,
   насколько разрыв объясняется незнанием слова самой LLM, а не потерями
   перевода.
3. **SigLIP2** (`google/siglip2-base-patch16-224`) — косинусная близость
   текстовых эмбеддингов казахского слова и английского глосса. Низкая
   близость на культурных словах — прямое предсказание, что Архитектура 3
   (казахский текст напрямую в карту ценности, без перевода) на них
   провалится ещё до того, как дело дойдёт до изображения.

Главная величина для статьи — разрыв «универсальные минус культурные»
слова и Cohen's d по каждому сигналу (уже считается в
`05_cultural_gap_summary.csv`).

Флаги `--skip-nllb` / `--skip-qolda` / `--skip-siglip` позволяют прогонять
сигналы по отдельности (например, сначала SigLIP2 на CPU, потом NLLB).

## Что дальше, после пилота

Пять проверок отвечают на пять конкретных архитектурных вопросов:

| Проверка | Отвечает на вопрос |
|---|---|
| #1 Латентность | Сколько вызовов Qolda влезает в бюджет одного эпизода в каждом режиме? |
| #2 JSON-парсинг | Можно ли доверять Qolda как «мозгу», разбирающему команду, без ручного парсера-заглушки? |
| #3 Верификация | Даёт ли сервер калиброванную P(да), или придётся платить self-consistency-семплированием за каждую проверку кандидата? |
| #4 Grounding | Нужен ли отдельный детектор (GroundingDINO), или Qolda может сама находить объект — четвёртая архитектура? |
| #5 Культурный разрыв | Насколько серьёзна проблема вообще, отдельно для NLLB-baseline, для LLM-знаний и для мультиязычного энкодера? |

Результаты этих пяти проверок — прямой вход в выбор между тремя
архитектурами полного конвейера (переводчик-baseline / Qolda-как-мозг /
SigLIP2 напрямую) и решение, нужно ли дообучение LoRA на культурных
объектах.

## Известные ограничения

- Автоматическая метрика `target_kk_exact_match_rate` в проверке #2 —
  точное совпадение строк после нормализации регистра/пробелов. Она
  жёстче, чем нужно (не засчитает синоним), поэтому годится как нижняя
  граница качества, не как окончательная метрика точности.
- `jaccard_token_overlap` в проверке #5 — простое лексическое перекрытие
  по токенам, а не семантическая близость. Для более строгой оценки
  перевода стоит добавить BLEU/chrF (например, через `sacrebleu`) или
  ручную экспертную разметку недостающих переводов.
- Вес гипотез (`w` в `ParsedCommand.hypotheses`) не откалиброван моделью —
  в дизайн-документе проекта явно указано считать его по частоте в
  нескольких сэмплах, а не доверять числу в JSON. Это ещё не
  реализовано отдельным скриптом; при необходимости добавляется поверх
  `qolda_pilot/client.py` (`n=k` в `chat()` уже поддерживается).
#   t e s t _ q o l d a _ v l m 
 
 
