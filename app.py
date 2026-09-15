import os
import tempfile
import cv2
import mido
import numpy as np
import streamlit as st

st.title("YTPMV Note-Flip Synchronizer")
st.write("選択したトラックのMIDIの音（ノート）ごとに、動画をパッと反転させます。")

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
        
        # トラックごとの名前や情報リストを作成
        track_options = []
        for i, track in enumerate(mid.tracks):
            track_name = track.name if track.name else f"Track {i}"
            track_options.append(f"{i}: {track_name} (イベント数: {len(track)})")

        selected_track_str = st.selectbox("反転させたいトラックを選択", track_options)
        selected_track_idx = int(selected_track_str.split(":")[0])

        # 演出スタイルの選択
        flip_style = st.radio(
            "反転のスタイル",
            [
                "音ごとに反転状態を切り替える（ON/OFFトグル）",
                "音が鳴った瞬間だけ左右反転してすぐに戻る（フラッシュ風）"
            ]
        )

        if st.button("YTPMV風動画を生成する"):
            with st.spinner("音ハメ動画を作成中..."):
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or np.isnan(fps):
                    fps = 30.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # 選択されたトラックの「音が鳴るタイミング（秒数）」を正確に抽出
                target_track = mid.tracks[selected_track_idx]
                note_times = []
                current_time = 0.0
                tempo = 500000  # 初期BPM 120相当

                for msg in target_track:
                    # ティックを秒数に変換
                    current_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
                    if msg.type == "set_tempo":
                        tempo = msg.tempo
                    elif msg.type == "note_on" and msg.velocity > 0:
                        note_times.append(current_time)

                output_path = "output_ytpmv.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                current_frame = 0
                note_idx = 0
                is_flipped = False

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    now_sec = current_frame / fps

                    # 現在のフレームの時間が、次のノートの時間を過ぎているかチェック
                    hit_note = False
                    while note_idx < len(note_times) and now_sec >= note_times[note_idx]:
                        hit_note = True
                        note_idx += 1

                    if hit_note:
                        if flip_style == "音ごとに反転状態を切り替える（ON/OFFトグル）":
                            is_flipped = not is_flipped
                        else:
                            # フラッシュ風の場合は一時的に反転を強制
                            is_flipped = True
                    else:
                        if flip_style == "音が鳴った瞬間だけ左右反転してすぐに戻る（フラッシュ風）":
                            # 次のフレームでは元に戻す
                            is_flipped = False

                    # 反転処理の適用
                    if is_flipped:
                        frame = cv2.flip(frame, 1) # 1は左右反転

                    out.write(frame)
                    current_frame += 1

                cap.release()
                out.release()

                st.success("生成が完了しました！")
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="動画をダウンロード",
                        data=f,
                        file_name="ytpmv_synced.mp4",
                        mime="video/mp4"
                    )

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

    finally:
        if os.path.exists(midi_path):
            os.remove(midi_path)
        if os.path.exists(video_path):
            os.remove(video_path)
