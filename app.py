import os
import tempfile

import streamlit as st
import mido
import cv2
import numpy as np


# =========================================================
# ページ設定
# =========================================================

st.set_page_config(
    page_title="YTPMV向けMIDI反転ツール",
    layout="centered"
)

st.title("YTPMV向けMIDI反転ツール")
st.write("スマホ勢向けの動画反転ツールです。パソコン勢の方もどうぞ。")


# =========================================================
# MIDI関連
# =========================================================

def get_track_name(track, index):
    name = None

    for msg in track:
        if msg.is_meta and msg.type == "track_name":
            name = msg.name
            break

    if name is None:
        return f"Track {index}"

    if isinstance(name, bytes):
        if len(name) == 0:
            return f"Track {index}"

        for encoding in ("utf-8", "cp932", "latin1"):
            try:
                decoded = name.decode(encoding).strip()

                if decoded:
                    return decoded

            except Exception:
                pass

        return f"Track {index}"

    name = str(name).strip()

    if not name or name in ("b''", 'b""'):
        return f"Track {index}"

    return name


def get_note_on_events(track):
    events = []

    for msg in track:
        if msg.type == "note_on":
            velocity = getattr(msg, "velocity", 0)

            if velocity > 0:
                events.append(msg)

    return events


def count_notes(track):
    return len(get_note_on_events(track))


def get_tempo_events(mid):
    tempo_events = []

    for track in mid.tracks:

        absolute_tick = 0

        for msg in track:

            absolute_tick += msg.time

            if msg.is_meta and msg.type == "set_tempo":

                tempo_events.append(
                    (absolute_tick, msg.tempo)
                )

    tempo_events.sort(key=lambda x: x[0])

    cleaned = []

    for tick, tempo in tempo_events:

        if cleaned and cleaned[-1][0] == tick:
            cleaned[-1] = (tick, tempo)

        else:
            cleaned.append(
                (tick, tempo)
            )

    if not cleaned or cleaned[0][0] > 0:

        cleaned.insert(
            0,
            (0, mido.bpm2tempo(120))
        )

    return cleaned


def tick_to_seconds(
    tick,
    ticks_per_beat,
    tempo_events
):

    if tick <= 0:
        return 0.0

    seconds = 0.0

    previous_tick = 0
    current_tempo = tempo_events[0][1]

    for tempo_tick, tempo in tempo_events:

        if tempo_tick > tick:
            break

        if tempo_tick > previous_tick:

            delta_ticks = (
                tempo_tick - previous_tick
            )

            seconds += (
                delta_ticks
                * current_tempo
                / 1_000_000.0
                / ticks_per_beat
            )

        previous_tick = tempo_tick
        current_tempo = tempo

    if tick > previous_tick:

        delta_ticks = (
            tick - previous_tick
        )

        seconds += (
            delta_ticks
            * current_tempo
            / 1_000_000.0
            / ticks_per_beat
        )

    return seconds


def get_note_times(mid, track_index):

    track = mid.tracks[track_index]

    tempo_events = get_tempo_events(mid)

    absolute_tick = 0
    note_times = []

    for msg in track:

        absolute_tick += msg.time

        if msg.type == "note_on":

            velocity = getattr(
                msg,
                "velocity",
                0
            )

            if velocity > 0:

                seconds = tick_to_seconds(
                    absolute_tick,
                    mid.ticks_per_beat,
                    tempo_events
                )

                note_times.append(seconds)

    if not note_times:
        return []

    # 最初のノートを0秒にする

    first_time = note_times[0]

    note_times = [
        max(0.0, t - first_time)
        for t in note_times
    ]

    # 同時発音は1つの切り替えとして扱う

    unique_times = []

    for t in note_times:

        if not unique_times:

            unique_times.append(t)

        elif abs(
            t - unique_times[-1]
        ) > 0.000001:

            unique_times.append(t)

    return unique_times


# =========================================================
# 動画関連
# =========================================================

def flip_frame(frame, mode):

    # 0 = 通常
    # 1 = 左右反転
    # 2 = 上下反転
    # 3 = 上下左右反転

    if mode == 0:
        return frame

    if mode == 1:
        return cv2.flip(frame, 1)

    if mode == 2:
        return cv2.flip(frame, 0)

    if mode == 3:
        return cv2.flip(frame, -1)

    return frame


def create_writer(
    output_path,
    fps,
    width,
    height
):

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            "出力動画を作成できませんでした。"
        )

    return writer


def read_source_frames(video_path):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            "動画を開けませんでした。"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(
            "動画からフレームを読み込めませんでした。"
        )

    height, width = frames[0].shape[:2]

    return (
        frames,
        fps,
        width,
        height
    )


# =========================================================
# ① 反転なし（ノートごと再生）
# =========================================================

def make_no_flip_video(
    video_path,
    note_times,
    output_path,
    progress_callback=None
):

    frames, fps, width, height = (
        read_source_frames(video_path)
    )

    writer = create_writer(
        output_path,
        fps,
        width,
        height
    )

    source_frame_count = len(frames)

    if not note_times:

        for i, frame in enumerate(frames):

            writer.write(frame)

            if progress_callback:

                progress_callback(
                    (i + 1)
                    / source_frame_count
                )

        writer.release()
        return

    for note_index in range(
        len(note_times)
    ):

        if (
            note_index + 1
            < len(note_times)
        ):

            duration = (
                note_times[note_index + 1]
                - note_times[note_index]
            )

        else:

            duration = (
                source_frame_count
                / fps
            )

        duration = max(
            0.0,
            duration
        )

        frame_count = max(
            1,
            int(round(
                duration * fps
            ))
        )

        for i in range(frame_count):

            source_index = i

            if (
                source_index
                >= source_frame_count
            ):
                break

            writer.write(
                frames[source_index]
            )

        if progress_callback:

            progress_callback(
                (note_index + 1)
                / len(note_times)
            )

    writer.release()


# =========================================================
# ② ノートごと反転
# =========================================================

def make_note_restart_video(
    video_path,
    note_times,
    output_path,
    progress_callback=None
):

    frames, fps, width, height = (
        read_source_frames(video_path)
    )

    writer = create_writer(
        output_path,
        fps,
        width,
        height
    )

    source_frame_count = len(frames)

    if not note_times:

        for frame in frames:
            writer.write(frame)

        writer.release()
        return

    for note_index in range(
        len(note_times)
    ):

        if (
            note_index + 1
            < len(note_times)
        ):

            duration = (
                note_times[note_index + 1]
                - note_times[note_index]
            )

        else:

            duration = (
                source_frame_count
                / fps
            )

        duration = max(
            0.0,
            duration
        )

        frame_count = max(
            1,
            int(round(
                duration * fps
            ))
        )

        # 1個目 通常
        # 2個目 左右反転
        # 3個目 通常
        # 4個目 左右反転...

        flip_mode = note_index % 2

        for i in range(frame_count):

            source_index = i

            if (
                source_index
                >= source_frame_count
            ):
                break

            frame = frames[
                source_index
            ]

            frame = flip_frame(
                frame,
                flip_mode
            )

            writer.write(frame)

        if progress_callback:

            progress_callback(
                (note_index + 1)
                / len(note_times)
            )

    writer.release()


# =========================================================
# ③ 動画そのまま反転
# =========================================================

def make_normal_video(
    video_path,
    note_times,
    output_path,
    progress_callback=None
):

    frames, fps, width, height = (
        read_source_frames(video_path)
    )

    writer = create_writer(
        output_path,
        fps,
        width,
        height
    )

    source_frame_count = len(frames)

    if not note_times:

        for i, frame in enumerate(frames):

            writer.write(frame)

            if progress_callback:

                progress_callback(
                    (i + 1)
                    / source_frame_count
                )

        writer.release()
        return

    video_duration = (
        source_frame_count
        / fps
    )

    last_note_time = note_times[-1]

    output_duration = max(
        video_duration,
        last_note_time
        + 1.0 / fps
    )

    output_frame_count = int(
        np.ceil(
            output_duration * fps
        )
    )

    note_index = 0

    for frame_index in range(
        output_frame_count
    ):

        current_time = (
            frame_index / fps
        )

        while (
            note_index + 1
            < len(note_times)
            and current_time
            >= note_times[note_index + 1]
        ):

            note_index += 1

        # 通常 → 左右反転 → 通常...

        flip_mode = note_index % 2

        if (
            frame_index
            < source_frame_count
        ):

            frame = frames[
                frame_index
            ]

        else:

            frame = frames[-1]

        frame = flip_frame(
            frame,
            flip_mode
        )

        writer.write(frame)

        if progress_callback:

            progress_callback(
                (frame_index + 1)
                / output_frame_count
            )

    writer.release()


# =========================================================
# ④ 上下左右反転
# =========================================================

def make_four_direction_video(
    video_path,
    note_times,
    output_path,
    progress_callback=None
):

    frames, fps, width, height = (
        read_source_frames(video_path)
    )

    writer = create_writer(
        output_path,
        fps,
        width,
        height
    )

    source_frame_count = len(frames)

    if not note_times:

        for i, frame in enumerate(frames):

            writer.write(frame)

            if progress_callback:

                progress_callback(
                    (i + 1)
                    / source_frame_count
                )

        writer.release()
        return

    flip_modes = [
        0,
        1,
        2,
        3
    ]

    for note_index in range(
        len(note_times)
    ):

        if (
            note_index + 1
            < len(note_times)
        ):

            duration = (
                note_times[note_index + 1]
                - note_times[note_index]
            )

        else:

            duration = (
                source_frame_count
                / fps
            )

        duration = max(
            0.0,
            duration
        )

        frame_count = max(
            1,
            int(round(
                duration * fps
            ))
        )

        flip_mode = (
            flip_modes[
                note_index % 4
            ]
        )

        for i in range(frame_count):

            source_index = i

            if (
                source_index
                >= source_frame_count
            ):
                break

            frame = frames[
                source_index
            ]

            frame = flip_frame(
                frame,
                flip_mode
            )

            writer.write(frame)

        if progress_callback:

            progress_callback(
                (note_index + 1)
                / len(note_times)
            )

    writer.release()


# =========================================================
# ① MIDI
# =========================================================

midi_file = st.file_uploader(
    "①MIDI",
    type=[
        "mid",
        "midi"
    ]
)


# =========================================================
# MIDI解析
# =========================================================

mid = None
midi_temp_path = None
selected_track_index = None


if midi_file is not None:

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mid"
        ) as tmp:

            tmp.write(
                midi_file.getbuffer()
            )

            midi_temp_path = tmp.name

        mid = mido.MidiFile(
            midi_temp_path
        )

        # =============================================
        # ② トラック
        # =============================================

        track_options = []

        for index, track in enumerate(
            mid.tracks
        ):

            note_count = count_notes(
                track
            )

            # 音数0のトラックは表示しない

            if note_count <= 0:
                continue

            track_name = get_track_name(
                track,
                index
            )

            label = (
                f"{index}: "
                f"{track_name}"
                f"（音数: {note_count}）"
            )

            track_options.append(
                (
                    index,
                    label,
                    note_count
                )
            )

        if track_options:

            st.subheader(
                "②トラック"
            )

            labels = [
                item[1]
                for item in track_options
            ]

            selected_label = st.selectbox(
                "使用するMIDIトラックを選択してください",
                labels
            )

            selected_track_index = next(
                item[0]
                for item in track_options
                if item[1] == selected_label
            )

        else:

            st.error(
                "ノートが入っているMIDIトラックが見つかりませんでした。"
            )

    except Exception as e:

        st.error(
            f"MIDIの読み込みに失敗しました：{e}"
        )


# =========================================================
# ③ 動画
# =========================================================

video_file = st.file_uploader(
    "③動画",
    type=[
        "mp4",
        "mov"
    ]
)


# =========================================================
# 動画プレビュー
# =========================================================

if video_file is not None:

    st.video(video_file)


# =========================================================
# ④ 反転
# =========================================================

st.subheader("④反転")

mode = st.radio(
    "反転方法を選択してください",
    [
        "反転なし（ノートごと再生）",
        "ノートごと反転",
        "動画そのまま反転",
        "上下左右反転"
    ]
)


# =========================================================
# 動画生成
# =========================================================

if (
    midi_file is not None
    and video_file is not None
    and mid is not None
    and selected_track_index is not None
):

    st.divider()

    if st.button(
        "🎬 動画を生成",
        type="primary"
    ):

        video_temp_path = None
        output_path = None

        try:

            # =========================================
            # 動画を一時保存
            # =========================================

            video_suffix = os.path.splitext(
                video_file.name
            )[1]

            if not video_suffix:
                video_suffix = ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=video_suffix
            ) as tmp:

                tmp.write(
                    video_file.getbuffer()
                )

                video_temp_path = tmp.name

            # =========================================
            # MIDIノート時刻
            # =========================================

            note_times = get_note_times(
                mid,
                selected_track_index
            )

            if not note_times:

                st.error(
                    "選択したトラックにノートがありません。"
                )

                st.stop()

            # =========================================
            # 出力先
            # =========================================

            output_path = os.path.join(
                tempfile.gettempdir(),
                "ytpmv_output.mp4"
            )

            progress_bar = st.progress(
                0
            )

            def update_progress(value):

                progress_bar.progress(
                    min(
                        1.0,
                        max(
                            0.0,
                            value
                        )
                    )
                )

            # =========================================
            # モード別生成
            # =========================================

            if mode == "反転なし（ノートごと再生）":

                make_no_flip_video(
                    video_temp_path,
                    note_times,
                    output_path,
                    update_progress
                )

            elif mode == "ノートごと反転":

                make_note_restart_video(
                    video_temp_path,
                    note_times,
                    output_path,
                    update_progress
                )

            elif mode == "動画そのまま反転":

                make_normal_video(
                    video_temp_path,
                    note_times,
                    output_path,
                    update_progress
                )

            elif mode == "上下左右反転":

                make_four_direction_video(
                    video_temp_path,
                    note_times,
                    output_path,
                    update_progress
                )

            progress_bar.progress(
                1.0
            )

            # =========================================
            # 完成
            # =========================================

            st.success(
                "🎉 動画が完成しました！"
            )

            st.video(
                output_path
            )

            with open(
                output_path,
                "rb"
            ) as f:

                st.download_button(
                    "⬇️ 完成した動画をダウンロード",
                    data=f,
                    file_name="ytpmv_output.mp4",
                    mime="video/mp4"
                )

        except Exception as e:

            st.error(
                f"動画生成中にエラーが発生しました：{e}"
            )

        finally:

            if video_temp_path:

                try:
                    os.remove(
                        video_temp_path
                    )
                except Exception:
                    pass

            if midi_temp_path:

                try:
                    os.remove(
                        midi_temp_path
                    )
                except Exception:
                    pass
