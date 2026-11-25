# Voice Conversion Studio

Desktop-приложение на Python (Windows) для преобразования голоса в реальном
времени: пользователь создаёт **voice profile** из своей аудиозаписи (или
записи, на использование которой у него есть право), а затем говорит в
микрофон и слышит на выходе речь с тембром выбранного профиля.

## Установка 

python -m venv .venv

.venv\Scripts\Activate.ps1

pip install torch --index-url 

pip install -r requirements.txt


pip install -r requirements-python314.txt
pip install --no-deps so-vits-svc-fork==4.2.30
pip install click cm-time fastapi lightning matplotlib pebble praat-parselmouth `
   psutil PySimpleGUI-4-foss torch torchaudio torchcrepe transformers `
   tqdm tqdm-joblib sounddevice tensorboard tensorboardx


Если на шаге 3 вы его пропустите, `pip install -r requirements.txt`
поставит CPU-версию `torch` — программа всё равно запустится и будет
работать, но в режиме CPU (см. предупреждение "GPU acceleration
unavailable. Using CPU mode.


