import os
import tempfile
import cv2
import mido
import numpy as np
import streamlit as st

st.title("YTPMV Note-Flip Synchronizer (Debug)")
st.write("映像の反転とコーデックを完全に安定させたバージョンです。")

midi_file = st.file_uploader("MIDIファイルをアップロード (.mid)", type=["mid", "midi"])
video_file = st.file_uploader("動画ファイルをアップロード (.mp4)", type=["mp4", "mov"])

if midi_file is not None and video_file is not None:
    st.success("ファイルが揃いました！")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
        f.write(midi_file.getvalue())
        midi_path = f.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(video_file.getvalue())
        video_path = f.name

    try:
        mid = mido.MidiFile(midi_path)
        
        track_options = []
        for i, track in enumerate(mid.tracks):
            track_name = track.name if track.name else f"Track {i}"
            track_options.append(f"{i}: {track_name} (イベント数: {len(track)})")

        selected_track_str = st.selectbox("反転させたいトラックを選択", track_options)
        selected_track_idx = int(selected_track_str.split(":")[0])

        flip_style = st.radio(
            "反転のスタイル",
            [
                "音ごとに反転状態をトグル切り替え（ON/OFF）",
                "音が鳴った瞬間から数フレーム間、強制反転（フラッシュ風）"
            ]
        )

        if st.button("動画を生成する"):
            with st.spinner("映像を解析・生成中..."):
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or np.isnan(fps):
                    fps = 30.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # MIDIの「音が鳴る秒数」を正確に抽出
                target_track = mid.tracks[selected_track_idx]
                note_times = []
                current_time = 0.0
                tempo = 500000

                for msg in target_track:
                    current_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
                    if msg.type == "set_tempo":
                        tempo = msg.tempo
                    elif msg.type == "note_on" and msg.velocity > 0:
                        note_times.append(current_time)

                st.write(f"検出された音の数: {len(note_times)} 個")

                output_path = "output_debug.mp4"
                
                # コーデックの互換性を上げるための設定
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # 万が一 VideoWriter が開かなかった場合のチェック
                if not out.isOpened():
                    st.error("動画ライターの初期化に失敗しました。")

                current_frame = 0
                note_idx = 0
                is_flipped = False
                flash_counter = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    now_sec = current_frame / fps

                    # 音のタイミングチェック
                    while note_idx < len(note_times) and now_sec >= note_times[note_idx]:
                        if flip_style == "音ごとに反転状態をトグル切り替え（ON/OFF）":
                            is_flipped = not is_flipped
                        else:
                            is_flipped = True
                            flash_counter = int(fps / 6)  # 約0.15秒間維持（分かりやすく長めに）
                        note_idx += 1

                    if flash_counter > 0:
                        flash_counter -= 1
                        if flash_counter == 0 and flip_style == "音が鳴った瞬間から数フレーム間、強制反転（フラッシュ風）":
                            is_flipped = False

                    # 1. 左右反転の実行
                    if is_flipped:
                        frame = cv2.flip(frame, 1)
                        # デバッグ用：反転しているフレームには画面上に赤文字で「FLASH!」を出す
                        cv2.putText(frame, "FLASH!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

                    out.write(frame)
                    current_frame += 1

                cap.release()
                out.release()

                st.success("生成が完了しました！")
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="完成した動画をダウンロード",
                        data=f,
                        file_name="ytpmv_debug.mp4",
                        mime="video/mp4"
                    )

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

    finally:
        if os.path.exists(midi_path):
            os.remove(midi_path)
        if os.path.exists(video_path):
            os.remove(video_path)
