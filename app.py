import os
import tempfile
import cv2
import mido
import numpy as np
import streamlit as st

st.title("YTPMV Note-Flip Synchronizer (Fixed)")
st.write("MIDIの音のタイミングに確実に合わせて動画を反転させます。")

midi_file = st.file_uploader("MIDIファイルをアップロード (.mid)", type=["mid", "midi"])
video_file = st.file_uploader("動画ファイルをアップロード (.mp4)", type=["mp4", "mov"])

if midi_file is not None and video_file is not None:
    st.success("ファイルが揃えました！")

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

                # --- MIDIの「音が鳴る秒数」を正確に抽出 ---
                target_track = mid.tracks[selected_track_idx]
                note_times = []
                current_time = 0.0
                tempo = 500000  # 初期BPM 120

                for msg in target_track:
                    current_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
                    if msg.type == "set_tempo":
                        tempo = msg.tempo
                    elif msg.type == "note_on" and msg.velocity > 0:
                        note_times.append(current_time)

                # デバッグ用に検出された音の数を表示
                st.write(f"このトラックで検出された音の数: {len(note_times)} 個")

                output_path = "output_ytpmv_fixed.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                current_frame = 0
                note_idx = 0
                is_flipped = False
                flash_counter = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    now_sec = current_frame / fps

                    # 音のタイミングに到達しているかチェック
                    while note_idx < len(note_times) and now_sec >= note_times[note_idx]:
                        if flip_style == "音ごとに反転状態をトグル切り替え（ON/OFF）":
                            is_flipped = not is_flipped
                        else:
                            # フラッシュ風：音が鳴ったら強制反転＋残りフレーム数をセット（例: 5フレーム分反転維持）
                            is_flipped = True
                            flash_counter = int(fps / 10)  # 約0.05秒間反転を維持
                        note_idx += 1

                    # フラッシュの残りフレームがある場合の処理
                    if flash_counter > 0:
                        flash_counter -= 1
                        if flash_counter == 0 and flip_style == "音が鳴った瞬間から数フレーム間、強制反転（フラッシュ風）":
                            is_flipped = False

                    # 実際にフレームを左右反転
                    if is_flipped:
                        frame = cv2.flip(frame, 1)

                    out.write(frame)
                    current_frame += 1

                cap.release()
                out.release()

                st.success("動画の生成が完了しました！")
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="完成した動画をダウンロード",
                        data=f,
                        file_name="ytpmv_fixed.mp4",
                        mime="video/mp4"
                    )

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

    finally:
        if os.path.exists(midi_path):
            os.remove(midi_path)
        if os.path.exists(video_path):
            os.remove(video_path)
