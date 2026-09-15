import os
import tempfile
import subprocess

import streamlit as st
import mido
import cv2
import numpy as np


# ============================================================
# ページ設定
# ============================================================

st.set_page_config(
    page_title="YTPMV MIDI Sync",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 YTPMV MIDI Sync")
st.write("MIDIの1音ごとに動画を左右反転します。")


# ============================================================
# MIDIの各トラックからノートの時間を取得
# ============================================================

def get_track_notes(midi, track_number):
    """
    選択したMIDIトラックのNote Onを取得。
    MIDI tickを秒に変換する。
    """

    track = midi.tracks[track_number]

    # MIDIでは通常500000μs = 120 BPM
    tempo = 500000

    current_tick = 0
    current_seconds = 0.0

    # そのトラックで発生するノート
    notes = []

    # テンポ変更を探す
    tempo_events = []

    for t in midi.tracks:

        tick = 0

        for msg in t:

            tick += msg.time

            if msg.type == "set_tempo":

                tempo_events.append(
                    (tick, msg.tempo)
                )

    tempo_events.sort(
        key=lambda x: x[0]
    )

    tempo_index = 0
    previous_tick = 0

    for msg in track:

        current_tick += msg.time

        # ----------------------------------------------------
        # 現在の位置までのテンポ変更を処理
        # ----------------------------------------------------

        while (
            tempo_index < len(tempo_events)
            and tempo_events[tempo_index][0]
            <= current_tick
        ):

            tempo_tick, new_tempo = (
                tempo_events[tempo_index]
            )

            delta = (
                tempo_tick
                - previous_tick
            )

            if delta > 0:

                current_seconds += (
                    mido.tick2second(
                        delta,
                        midi.ticks_per_beat,
                        tempo
                    )
                )

            previous_tick = tempo_tick
            tempo = new_tempo

            tempo_index += 1

        # ----------------------------------------------------
        # 残りのtickを秒に変換
        # ----------------------------------------------------

        delta = (
            current_tick
            - previous_tick
        )

        if delta > 0:

            current_seconds += (
                mido.tick2second(
                    delta,
                    midi.ticks_per_beat,
                    tempo
                )
            )

            previous_tick = current_tick

        # ----------------------------------------------------
        # Note On
        # ----------------------------------------------------

        if (
            msg.type == "note_on"
            and msg.velocity > 0
        ):

            notes.append({
                "time": current_seconds,
                "note": msg.note,
                "velocity": msg.velocity
            })

    return notes


# ============================================================
# 動画情報
# ============================================================

def get_video_info(path):

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():

        raise RuntimeError(
            "動画を開けませんでした。"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    duration = (
        frames / fps
        if fps > 0
        else 0
    )

    cap.release()

    return {
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "duration": duration
    }


# ============================================================
# 動画生成
# ============================================================

def make_video(
    input_video,
    output_video,
    notes
):

    cap = cv2.VideoCapture(
        input_video
    )

    if not cap.isOpened():

        raise RuntimeError(
            "動画を開けませんでした。"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        raise RuntimeError(
            "FPSを取得できませんでした。"
        )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    # --------------------------------------------------------
    # MIDIノート時刻
    # --------------------------------------------------------

    note_times = np.array(
        [
            float(n["time"])
            for n in notes
        ],
        dtype=np.float64
    )

    note_times.sort()

    # --------------------------------------------------------
    # 一時動画
    # --------------------------------------------------------

    temp_video = (
        output_video
        + ".video.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        temp_video,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "動画を書き込めませんでした。"
        )

    progress = st.progress(0)

    # ========================================================
    # フレーム処理
    # ========================================================

    for frame_number in range(
        frame_count
    ):

        success, frame = cap.read()

        if not success:
            break

        # ----------------------------------------------------
        # フレームの時間
        # ----------------------------------------------------

        current_time = (
            frame_number / fps
        )

        # ----------------------------------------------------
        # この時刻までに何音鳴ったか
        # ----------------------------------------------------

        notes_before = np.searchsorted(
            note_times,
            current_time,
            side="right"
        )

        # ----------------------------------------------------
        # 1音ごとに状態を切り替える
        #
        # 0音 → 通常
        # 1音 → 左右反転
        # 2音 → 通常
        # 3音 → 左右反転
        # 4音 → 通常
        # ...
        # ----------------------------------------------------

        if notes_before % 2 == 1:

            frame = cv2.flip(
                frame,
                1
            )

        # ----------------------------------------------------
        # フレーム保存
        # ----------------------------------------------------

        writer.write(
            frame
        )

        # ----------------------------------------------------
        # 進捗
        # ----------------------------------------------------

        if frame_number % 10 == 0:

            progress.progress(
                min(
                    frame_number
                    / max(frame_count, 1),
                    1.0
                )
            )

    cap.release()
    writer.release()

    progress.progress(1.0)

    # ========================================================
    # FFmpegで元動画の音声を戻す
    # ========================================================

    command = [
        "ffmpeg",
        "-y",

        "-i",
        temp_video,

        "-i",
        input_video,

        "-map",
        "0:v:0",

        "-map",
        "1:a?",

        "-c:v",
        "libx264",

        "-preset",
        "veryfast",

        "-crf",
        "18",

        "-pix_fmt",
        "yuv420p",

        "-c:a",
        "aac",

        "-b:a",
        "192k",

        "-shortest",

        output_video
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if os.path.exists(
        temp_video
    ):

        os.remove(
            temp_video
        )

    if result.returncode != 0:

        raise RuntimeError(
            "FFmpegエラー:\n"
            + result.stderr[-5000:]
        )


# ============================================================
# MIDIアップロード
# ============================================================

st.header("① MIDI")

midi_file = st.file_uploader(
    "MIDIファイルをアップロード",
    type=[
        "mid",
        "midi"
    ]
)

midi = None
midi_path = None
notes = []


if midi_file:

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mid"
        ) as f:

            f.write(
                midi_file.getvalue()
            )

            midi_path = f.name

        midi = mido.MidiFile(
            midi_path
        )

        # ----------------------------------------------------
        # トラック一覧
        # ----------------------------------------------------

        track_names = []

        for i, track in enumerate(
            midi.tracks
        ):

            name = ""

            for msg in track:

                if msg.type == "track_name":

                    name = msg.name
                    break

            if not name:

                name = f"Track {i}"

            # ノート数
            note_count = sum(
                1
                for msg in track
                if (
                    msg.type == "note_on"
                    and msg.velocity > 0
                )
            )

            track_names.append(
                f"Track {i} - "
                f"{name} "
                f"({note_count} notes)"
            )

        st.success(
            "MIDIを読み込みました。"
        )

        # ----------------------------------------------------
        # トラック選択
        # ----------------------------------------------------

        selected_track = st.selectbox(
            "同期するMIDIトラック",
            range(
                len(track_names)
            ),
            format_func=lambda i:
                track_names[i]
        )

        # ----------------------------------------------------
        # ノート取得
        # ----------------------------------------------------

        notes = get_track_notes(
            midi,
            selected_track
        )

        st.write(
            f"🎵 ノート数：**{len(notes)}**"
        )

        # ノート時刻表示
        if notes:

            st.write(
                "最初の20音のタイミング（秒）:"
            )

            st.code(
                "\n".join(
                    [
                        f"{i + 1}: "
                        f"{n['time']:.6f} 秒 "
                        f"(Note {n['note']})"
                        for i, n
                        in enumerate(
                            notes[:20]
                        )
                    ]
                )
            )

        else:

            st.warning(
                "このトラックにはNote Onがありません。"
            )

    except Exception as e:

        st.error(
            "MIDI読み込みエラー："
            + str(e)
        )


# ============================================================
# 動画アップロード
# ============================================================

st.header("② 動画")

video_file = st.file_uploader(
    "動画ファイルをアップロード",
    type=[
        "mp4",
        "mov"
    ]
)

video_path = None
video_info = None


if video_file:

    try:

        extension = os.path.splitext(
            video_file.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as f:

            f.write(
                video_file.getvalue()
            )

            video_path = f.name

        video_info = get_video_info(
            video_path
        )

        st.success(
            "動画を読み込みました。"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "FPS",
            f"{video_info['fps']:.2f}"
        )

        col2.metric(
            "フレーム数",
            video_info["frames"]
        )

        col3.metric(
            "長さ",
            f"{video_info['duration']:.2f} 秒"
        )

    except Exception as e:

        st.error(
            "動画読み込みエラー："
            + str(e)
        )


# ============================================================
# 説明
# ============================================================

st.header("③ 同期方法")

st.info(
    """
MIDIのノートが鳴るたびに映像の状態を切り替えます。

1音目 → 左右反転
2音目 → 通常
3音目 → 左右反転
4音目 → 通常
5音目 → 左右反転
...
"""
)


# ============================================================
# 生成ボタン
# ============================================================

st.header("④ 生成")

if st.button(
    "🎬 MIDIに合わせて左右反転",
    type="primary",
    use_container_width=True
):

    # MIDIチェック
    if midi is None:

        st.error(
            "先にMIDIをアップロードしてください。"
        )

        st.stop()

    # ノートチェック
    if len(notes) == 0:

        st.error(
            "ノートがあるMIDIトラックを選択してください。"
        )

        st.stop()

    # 動画チェック
    if video_path is None:

        st.error(
            "先に動画をアップロードしてください。"
        )

        st.stop()

    # --------------------------------------------------------
    # 動画生成
    # --------------------------------------------------------

    output_path = os.path.join(
        tempfile.gettempdir(),
        "ytpmv_output.mp4"
    )

    try:

        with st.spinner(
            "MIDIに合わせて動画を処理しています..."
        ):

            make_video(
                video_path,
                output_path,
                notes
            )

        st.success(
            "🎉 完成しました！"
        )

        # ----------------------------------------------------
        # プレビュー
        # ----------------------------------------------------

        st.video(
            output_path
        )

        # ----------------------------------------------------
        # ダウンロード
        # ----------------------------------------------------

        with open(
            output_path,
            "rb"
        ) as f:

            st.download_button(
                "⬇️ 完成した動画を保存",
                data=f,
                file_name="ytpmv_output.mp4",
                mime="video/mp4",
                use_container_width=True
            )

    except Exception as e:

        st.error(
            "動画生成エラー"
        )

        st.code(
            str(e)
        )
