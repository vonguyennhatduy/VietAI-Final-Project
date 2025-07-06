FROM python:3.11-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./main.py /code/main.py
COPY ./logistic_final_titanic.pkl /code/logistic_final_titanic.pkl

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# FROM python:3.11-slim

# WORKDIR /code

# COPY ./requirements.txt /code/requirements.txt
# RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# COPY ./main.py /code/main.py
# COPY ./logistic_final_titanic.pkl /code/logistic_final_titanic.pkl

# # Đặt biến môi trường PORT mặc định, cho phép Render ghi đè
# ENV PORT=10000
# EXPOSE $PORT

# # Chạy Uvicorn với cổng từ biến môi trường
# CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 120"]