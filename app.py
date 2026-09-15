import os
import tempfile
import cv2
import mido
import numpy as np
import streamlit as st

st.title("MIDI Video Synchronizer (Fix)")
st.write("MIDIの音のタイミングに完全に合わせて、動画を同期・反転させます。")

# 1. ファイルのアップロード
midi_file = st.file_uploader(
    "1. MIDIファイルをアップロード (.mid)", type=["mid", "midi"]
)
video_file = st.file_uploader(
    "2. 動画ファイルをアップロード (mp4)", type=["mp4", "mov"]
)

if midi_file is not None and video_file is not None:
    st.success("ファイルが揃いました！")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
        f.write(midi_file.getvalue())
        midi_path = f.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(video_file.getvalue())
        video_path = f.name

    try:
        # MIDIファイルの読み込み
        mid = mido.MidiFile(midi_path)
        track_names = [
            f"Track {i}: {track.name}" for i, track in enumerate(mid.tracks)
        ]

        selected_track_idx = st.selectbox(
            "3. 使用するトラックを選択してください",
            range(len(track_names)),
            format_func=lambda x: track_names[x],
        )

        st.subheader("4. 演出パターンの設定")
        play_mode = st.radio(
            "動作モードを選んでください",
            [
                "反転だけして繰り返しはしない（音のタイミングで1回反転）",
                "1音ごとに動画を切り替え・フラッシュ（ループ再生風）",
            ],
        )

        if st.button("動画を生成する"):
            with st.spinner("MIDIと動画を精密に同期中..."):

                # 動画の基本情報を取得
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0 or np.isnan(fps):
                    fps = 30.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # --- MIDIの「音が鳴る秒数」をすべて抽出 ---
                target_track = mid.tracks[selected_track_idx]
                note_times = []  # 音が鳴るタイミング（秒）のリスト
                current_time = 0.0

                # テンポ（BPM）の追跡（初期値は120BPM = 500000 microseconds per beat）
                tempo = 500000

                for msg in target_track:
                    # デルタタイム（前のイベントからの経過ティック）を秒数に変換
                    current_time += mido.tick2second(
                        msg.time, mid.ticks_per_beat, tempo
                    )

                    if msg.type == "set_tempo":
                        tempo = msg.tempo
                    elif msg.type == "note_on" and msg.velocity > 0:
                        # 音が鳴った瞬間の「秒数」を記録
                        note_times.append(current_time)

                output_path = "output_precise.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                # 映像の編集ループ
                current_frame = 0
                is_flipped = False
                note_index = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    now_sec = current_frame / fps

                    # 次のノート（音）のタイミングに到達したかチェック
                    if (
                        note_index < len(note_times)
                        and now_sec >= note_times[note_index]
                    ):
                        if (
                            play_mode
                            == "1音ごとに動画を切り替え・フラッシュ（ループ再生風）"
                        ):
                            # 1音鳴るごとに動画の再生位置を最初に戻す、または激しく反転
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            current_frame = 0
                        else:
                            # 鳴るたびに反転状態をトグル（ON/OFF）切り替え
                            is_flipped = not is_flipped

                        note_index += 1

                    # 反転モードの場合の処理
                    if (
                        play_mode
                        == "反転だけして繰り返しはしない（音のタイミングで1回反転）"
                        and is_flipped
                    ):
                        frame = cv2.flip(frame, 1)

                    out.write(frame)
                    current_frame += 1

                cap.release()
                out.release()

                st.success("同期動画の生成が完了しました！")
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="完成した動画をダウンロード",
                        data=f,
                        file_name="midi_synced_perfect.mp4",
                        mime="video/mp4",
                    )

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

    finally:
        if os.path.exists(midi_path):
            os.remove(midi_path)
        if os.path.exists(video_path):
            os.remove(video_path)
