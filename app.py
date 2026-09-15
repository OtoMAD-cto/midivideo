import os
import tempfile
import cv2
import mido
import numpy as np
import streamlit as st

st.title("MIDI Video Synchronizer")
st.write(
    "MIDIのノート（音）に合わせて、動画を反転させたり繰り返したりするアプリです。"
)

# 1. MIDIファイルのアップロード
midi_file = st.file_uploader(
    "1. MIDIファイルをアップロード (.mid)", type=["mid", "midi"]
)

# 2. 動画ファイルのアップロード (mp4推奨)
video_file = st.file_uploader(
    "2. 背景や演出に使う動画をアップロード (mp4)", type=["mp4", "mov"]
)

if midi_file is not None and video_file is not None:
    st.success("ファイルが揃いました！")

    # 一時ファイルとして保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
        f.write(midi_file.getvalue())
        midi_path = f.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(video_file.getvalue())
        video_path = f.name

    try:
        # MIDIのトラック解析
        mid = mido.MidiFile(midi_path)
        track_names = [
            f"Track {i}: {track.name}" for i, track in enumerate(mid.tracks)
        ]

        # 3. トラック選択機能
        selected_track_idx = st.selectbox(
            "3. 使用するトラックを選択してください",
            range(len(track_names)),
            format_func=lambda x: track_names[x],
        )

        # 4. 反転・繰り返しのパターン選択
        st.subheader("4. 演出パターンの設定")
        play_mode = st.radio(
            "動作モードを選んでください",
            [
                "反転だけして繰り返しはしない",
                "1音ずつ繰り返すパターン (ループ/フラッシュ)",
            ],
        )

        if st.button("動画を生成する"):
            with st.spinner(
                "動画を処理中...（数秒〜数十秒かかります）"
            ):
                # 動画の読み込み
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0 or np.isnan(fps):
                    fps = 30  
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                output_path = "output_synced.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                # 簡単なフレーム処理（デモ用）
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 選択されたモードに応じた反転処理の例
                    if (
                        play_mode
                        == "1音ずつ繰り返すパターン (ループ/フラッシュ)"
                    ):
                        if (
                            int(frame_count / (fps / 4)) % 2 == 0
                        ):  
                            frame = cv2.flip(frame, 1)
                    else:
                        if frame_count > (cap.get(cv2.CAP_PROP_FRAME_COUNT) / 2):
                            frame = cv2.flip(frame, 1)

                    out.write(frame)
                    frame_count += 1

                cap.release()
                out.release()

                st.success("動画の生成が完了しました！")
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="完成した動画をダウンロード",
                        data=f,
                        file_name="midi_synced_output.mp4",
                        mime="video/mp4",
                    )

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

    finally:
        # 一時ファイルの削除
        if os.path.exists(midi_path):
            os.remove(midi_path)
        if os.path.exists(video_path):
            os.remove(video_path)
