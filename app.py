import os
import tempfile
import cv2
import mido
import numpy as np
import streamlit as st

st.title("MIDI Note Flipper")
st.write("MIDIの音（全トラック）に合わせて、音が鳴るたびに映像が反転します！")

midi_file = st.file_uploader("MIDIファイルをアップロード (.mid)", type=["mid", "midi"])
video_file = st.file_uploader("動画ファイルをアップロード (mp4)", type=["mp4", "mov"])

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

        if st.button("動画を生成する"):
            with st.spinner("音のタイミングを解析して反転動画を作成中..."):
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or np.isnan(fps):
                    fps = 30.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # ★すべてのトラックから「音が鳴るタイミング（秒）」を漏れなく集める
                note_times = []
                for track in mid.tracks:
                    current_time = 0.0
                    tempo = 500000
                    for msg in track:
                        current_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
                        if msg.type == "set_tempo":
                            tempo = msg.tempo
                        elif msg.type == "note_on" and msg.velocity > 0:
                            note_times.append(current_time)

                # 時間順に綺麗に並べ替える
                note_times.sort()

                if len(note_times) == 0:
                    st.warning("警告: このMIDIファイルから音のデータが見つかりませんでした。別のMIDIでお試しください。")

                output_path = "output_flipped.mp4"
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

                    # 音が鳴るタイミングが来たら、反転状態をパッと切り替える（トグル）
                    while note_idx < len(note_times) and now_sec >= note_times[note_idx]:
                        is_flipped = not is_flipped
                        note_idx += 1

                    # 反転フラグがONの間は左右反転
                    if is_flipped:
                        frame = cv2.flip(frame, 1)

                    out.write(frame)
                    current_frame += 1

                cap.release()
                out.release()

                st.success("反転動画の生成が完了しました！")
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="動画をダウンロード",
                        data=f,
                        file_name="note_flipped_output.mp4",
                        mime="video/mp4"
                    )

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
    finally:
        if os.path.exists(midi_path):
            os.remove(midi_path)
        if os.path.exists(video_path):
            os.remove(video_path)
