* mediapipe 환경 필수

1. #### 데이터 전처리(xml -> npy) 폴더명 참고

   **a_make_dataset.py**

```
python a_make_dataset.py
 --video_root "C:\Users\leesc\Downloads\Sample\01.원천데이터\03.이상행동"
 --xml_root   "C:\Users\leesc\Downloads\Sample\02.라벨링데이터\03.이상행동"
 --out_dir ds_lstm_all
 --win 16 --stride 4 --overlap 0.10 --resize_w 640 --model_complexity 0
```

2. #### 훈련(출력 폴더 및 epoch, 저장파일명 설정)

   **b_train_lstm.py**

```
python b_train_lstm.py --data_dir ds_lstm_all --epochs 30 --batch 64 --lr 5e-4 --save lstm_multilabel.pt
```

3. #### 실시간 추론(화면 보기)

   **c_realtime_inference_lstm.py**

```
python -u c_realtime_inference_lstm.py --ckpt lstm_multilabel.pt --src "C_3_10_1_BU_SMA_09-20_14-08-01_CB_RGB_DF2_M2.mp4"
```

4. #### 실시간 추론 후 서버 전송 클라이언트

   **c_realtime_inference_lstm.py**

```
python c_realtime_client.py \
  --ckpt lstm_multilabel.pt \
  --src 'C_3_9_1_BU_DYA_08-02_13-36-38_CA_RGB_DF2_M2.mp4' \
  --api_url http://127.0.0.1:8001/events \
  --start_thr 0.8 --end_thr 0.55 \
  --vote_win 5 --min_start_votes 4 --min_end_votes 4 \
  --min_event_sec 2.0 --cooldown_sec 3.0 \
  --throttle \
  --event_only \
  --print_probs \
  --log_prefix RUN
# HEARTBEAT까지 끄려면 --heartbeat_sec 0 도 함께 추가
```


python c_realtime_client.py --ckpt lstm_multilabel_05.pt --src 0 --api_url http://127.0.0.1:8001/events --start_thr 0.8 --end_thr 0.55 --vote_win 5 --min_start_votes 4 --min_end_votes 4 --min_event_sec 2.0 --cooldown_sec 3.0 --throttle --event_only --print_probs --log_prefix RUN

5. #### 실시간 추론 후 서버 전송 클라이언트

   **c_server.py**

```
python c_server.py
# 또는
uvicorn c_server:app --host 0.0.0.0 --port 8000 --reload
```


6. #### 임시로 만들어 본 _app.py , index.html

   **stream_lstm_app.py**
   **index.html**

```
lstm_multilabel_05.pt 필요
.env 필요 (.example 참고)
uvicorn stream_lstm_app:app --host 0.0.0.0 --port 8000 --reload
브라우저로 localhost:8000 접속
```