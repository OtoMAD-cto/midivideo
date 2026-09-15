import os
import tempfile
import subprocess

import streamlit as st
import mido
import cv2
import numpy as np


# ============================================================
# 設定
# ============================================================

st.set_page_config(
    page_title="YTPMV MIDI Sync",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 YTPMV MIDI Sync")
st.write("MIDIのノートに合わせて動画を左右反転します。")


# ============================================================
# MIDI読み込み
# ============================================================

def load_midi(file_data):

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mid"
    )

    temp.write(file_data)
    temp.close()

    midi = mido.MidiFile(temp.name)

    os.remove(temp.name)

    return midi


# ============================================================
# MIDIトラックのtick位置を取得
# ============================================================

def get_track_note_ticks(midi, track_index):

    track = midi.tracks[track_index]

    tick = 0
    notes = []

    for msg in track:

        tick += msg.time

        if (
            msg.type == "note_on"
            and msg.velocity > 0
        ):

            notes.append({
                "tick": tick,
                "note": msg.note,
                "velocity": msg.velocity
            })

    return notes


# ============================================================
# MIDI全体のテンポ情報
# ============================================================

def get_tempo_events(midi):

    events = []

    for track in midi.tracks:

        tick = 0

        for msg in track:

            tick += msg.time

            if msg.type == "set_tempo":

                events.append(
                    (tick, msg.tempo)
                )

    # 同じtickのものを整理
    events.sort(
        key=lambda x: x[0]
    )

    return events


# ============================================================
# tick → 秒
# ============================================================

def tick_to_seconds(
    midi,
    target_tick,
    tempo_events
):

    if target_tick <= 0:
        return 0.0

    # デフォルト120BPM
    tempo = 500000

    previous_tick = 0
    seconds = 0.0

    for tempo_tick, new_tempo in tempo_events:

        if tempo_tick > target_tick:
            break

        delta_tick = (
            tempo_tick
            - previous_tick
        )

        if delta_tick > 0:

            seconds += mido.tick2second(
                delta_tick,
                midi.ticks_per_beat,
                tempo
            )

        previous_tick = tempo_tick

        tempo = new_tempo

    # 最後のテンポから目的位置まで
    delta_tick = (
        target_tick
        - previous_tick
    )

    if delta_tick > 0:

        seconds += mido.tick2second(
            delta_tick,
            midi.ticks_per_beat,
            tempo
        )

    return seconds


# ============================================================
# MIDIノートを秒に変換
# ============================================================

def get_notes_in_seconds(
    midi,
    track_index
):

    note_ticks = get_track_note_ticks(
        midi,
        track_index
    )

    tempo_events = get_tempo_events(
        midi
    )

    notes = []

    for note in note_ticks:

        seconds = tick_to_seconds(
            midi,
            note["tick"],
            tempo_events
        )

        notes.append({
            "time": seconds,
            "note": note["note"],
            "velocity": note["velocity"]
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
# 動画を全部左右反転
# ============================================================

def flip_entire_video(
    input_path,
    output_path
):

    cap = cv2.VideoCapture(
        input_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            "動画を開けませんでした。"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
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

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    temp_video = (
        output_path
        + ".temp.mp4"
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
            "動画出力を開始できませんでした。"
        )

    progress = st.progress(0)

    for i in range(frame_count):

        ret, frame = cap.read()

        if not ret:
            break

        # ★ 確実に左右反転
        frame = cv2.flip(
            frame,
            1
        )

        writer.write(frame)

        if i % 10 == 0:

            progress.progress(
                min(
                    i / max(frame_count, 1),
                    1.0
                )
            )

    cap.release()
    writer.release()

    progress.progress(1.0)

    add_audio(
        temp_video,
        input_path,
        output_path
    )

    if os.path.exists(temp_video):

        os.remove(
            temp_video
        )


# ============================================================
# MIDI同期動画
# ============================================================

def make_midi_video(
    input_path,
    output_path,
    notes
):

    cap = cv2.VideoCapture(
        input_path
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
            "動画のFPSを取得できませんでした。"
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

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # MIDIノートの時間
    note_times = np.array(
        [
            n["time"]
            for n in notes
        ],
        dtype=np.float64
    )

    note_times.sort()

    temp_video = (
        output_path
        + ".temp.mp4"
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
            "動画出力を開始できませんでした。"
        )

    progress = st.progress(0)

    flipped_frames = 0

    # ========================================================
    # フレーム処理
    # ========================================================

    for frame_number in range(
        frame_count
    ):

        ret, frame = cap.read()

        if not ret:
            break

        # このフレームの時間
        current_time = (
            frame_number / fps
        )

        # ----------------------------------------------------
        # 現在時刻までに鳴った音の数
        # ----------------------------------------------------

        note_count = np.searchsorted(
            note_times,
            current_time,
            side="right"
        )

        # ----------------------------------------------------
        # 奇数音なら左右反転
        # ----------------------------------------------------

        should_flip = (
            note_count % 2 == 1
        )

        if should_flip:

            # ★ 左右反転
            frame = cv2.flip(
                frame,
                1
            )

            flipped_frames += 1

        writer.write(
            frame
        )

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
    # 音声追加
    # ========================================================

    add_audio(
        temp_video,
        input_path,
        output_path
    )

    if os.path.exists(temp_video):

        os.remove(
            temp_video
        )

    return flipped_frames


# ============================================================
# 元動画の音声を追加
# ============================================================

def add_audio(
    video_path,
    original_path,
    output_path
):

    command = [
        "ffmpeg",
        "-y",

        "-i",
        video_path,

        "-i",
        original_path,

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

        output_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            "FFmpegエラー:\n"
            + result.stderr[-4000:]
        )


# ============================================================
# ① MIDI
# ============================================================

st.header("① MIDIファイル")

midi_file = st.file_uploader(
    "MIDIを選択",
    type=["mid", "midi"]
)

midi = None
notes = []


if midi_file:

    try:

        midi = load_midi(
            midi_file.getvalue()
        )

        st.success(
            "MIDI読み込み成功"
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

            note_count = sum(
                1
                for msg in track
                if (
                    msg.type == "note_on"
                    and msg.velocity > 0
                )
            )

            track_names.append(
                f"Track {i} / "
                f"{name} / "
                f"{note_count} notes"
            )

        selected_track = st.selectbox(
            "使用するMIDIトラック",
            range(len(track_names)),
            format_func=lambda x:
                track_names[x]
        )

        notes = get_notes_in_seconds(
            midi,
            selected_track
        )

        st.write(
            f"ノート数：**{len(notes)}**"
        )

        # ----------------------------------------------------
        # ノート一覧
        # ----------------------------------------------------

        if notes:

            st.write(
                "最初の20音："
            )

            for i, note in enumerate(
                notes[:20]
            ):

                st.write(
                    f"{i + 1}音目："
                    f" {note['time']:.6f}秒"
                    f" / Note {note['note']}"
                )

        else:

            st.warning(
                "このトラックには音がありません。"
            )

    except Exception as e:

        st.error(
            f"MIDIエラー：{e}"
        )


# ============================================================
# ② 動画
# ============================================================

st.header("② 動画ファイル")

video_file = st.file_uploader(
    "動画を選択",
    type=["mp4", "mov"]
)

video_path = None


if video_file:

    try:

        extension = os.path.splitext(
            video_file.name
        )[1]

        temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        )

        temp.write(
            video_file.getvalue()
        )

        temp.close()

        video_path = temp.name

        info = get_video_info(
            video_path
        )

        st.success(
            "動画読み込み成功"
        )

        st.write(
            f"FPS：{info['fps']:.2f}"
        )

        st.write(
            f"長さ：{info['duration']:.2f}秒"
        )

        st.write(
            f"解像度："
            f"{info['width']} × "
            f"{info['height']}"
        )

    except Exception as e:

        st.error(
            f"動画エラー：{e}"
        )


# ============================================================
# ③ テスト
# ============================================================

st.header("③ 左右反転テスト")

st.write(
    "まずMIDIを使わず、動画そのものを左右反転できるか確認できます。"
)

if st.button(
    "🔄 動画全体を左右反転してテスト",
    use_container_width=True
):

    if video_path is None:

        st.error(
            "先に動画をアップロードしてください。"
        )

    else:

        output = os.path.join(
            tempfile.gettempdir(),
            "flip_test.mp4"
        )

        try:

            with st.spinner(
                "動画全体を左右反転しています..."
            ):

                flip_entire_video(
                    video_path,
                    output
                )

            st.success(
                "左右反転テスト完了"
            )

            st.video(
                output
            )

            with open(
                output,
                "rb"
            ) as f:

                st.download_button(
                    "⬇️ テスト動画を保存",
                    f,
                    file_name="flip_test.mp4",
                    mime="video/mp4"
                )

        except Exception as e:

            st.error(
                f"反転テストエラー：{e}"
            )


# ============================================================
# ④ MIDI同期
# ============================================================

st.header("④ MIDIに合わせて左右反転")

st.info(
    """
MIDIの音が鳴るたびに状態を切り替えます。

1音目 → 左右反転
2音目 → 通常
3音目 → 左右反転
4音目 → 通常
5音目 → 左右反転
...
"""
)


if st.button(
    "🎬 MIDI同期動画を生成",
    type="primary",
    use_container_width=True
):

    if midi is None:

        st.error(
            "MIDIをアップロードしてください。"
        )

        st.stop()

    if len(notes) == 0:

        st.error(
            "音が入っているトラックを選択してください。"
        )

        st.stop()

    if video_path is None:

        st.error(
            "動画をアップロードしてください。"
        )

        st.stop()

    output = os.path.join(
        tempfile.gettempdir(),
        "ytpmv_output.mp4"
    )

    try:

        with st.spinner(
            "MIDIに合わせて処理しています..."
        ):

            flipped = make_midi_video(
                video_path,
                output,
                notes
            )

        st.success(
            "🎉 完成しました！"
        )

        # ----------------------------------------------------
        # デバッグ情報
        # ----------------------------------------------------

        st.write(
            f"左右反転したフレーム数："
            f"**{flipped}**"
        )

        st.write(
            f"MIDIノート数："
            f"**{len(notes)}**"
        )

        st.video(
            output
        )

        with open(
            output,
            "rb"
        ) as f:

            st.download_button(
                "⬇️ 完成した動画を保存",
                f,
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
