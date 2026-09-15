import os
import tempfile
import streamlit as st
import mido
import cv2
import numpy as np


# =========================
# MIDI読み込み
# =========================

def load_midi(file_path):
    return mido.MidiFile(file_path)


# =========================
# MIDIのテンポ情報を取得
# =========================

def get_tempo_events(mid):
    tempo_events = []

    for track in mid.tracks:
        absolute_tick = 0

        for msg in track:
            absolute_tick += msg.time

            if msg.type == "set_tempo":
                tempo_events.append(
                    (absolute_tick, msg.tempo)
                )

    # tick順に並べる
    tempo_events.sort(key=lambda x: x[0])

    # 同じtickに複数ある場合は最後のものを使用
    result = []

    for tick, tempo in tempo_events:
        if result and result[-1][0] == tick:
            result[-1] = (tick, tempo)
        else:
            result.append((tick, tempo))

    # テンポ情報が無い場合は120BPM
    if not result:
        result = [(0, 500000)]

    # 0tickにテンポが無い場合
    if result[0][0] != 0:
        result.insert(0, (0, 500000))

    return result


# =========================
# MIDI tick → 秒
# =========================

def tick_to_seconds(target_tick, ticks_per_beat, tempo_events):

    seconds = 0.0
    previous_tick = 0
    current_tempo = 500000

    for tick, tempo in tempo_events:

        if tick > target_tick:
            break

        delta_tick = tick - previous_tick

        seconds += (
            delta_tick
            * current_tempo
            / 1_000_000
            / ticks_per_beat
        )

        previous_tick = tick
        current_tempo = tempo

    # 最後のテンポ区間
    delta_tick = target_tick - previous_tick

    seconds += (
        delta_tick
        * current_tempo
        / 1_000_000
        / ticks_per_beat
    )

    return seconds


# =========================
# 選択トラックのノート取得
# =========================

def get_notes_in_seconds(mid, track_index):

    track = mid.tracks[track_index]

    tempo_events = get_tempo_events(mid)

    absolute_tick = 0
    notes = []

    for msg in track:

        absolute_tick += msg.time

        if (
            msg.type == "note_on"
            and msg.velocity > 0
        ):

            time_sec = tick_to_seconds(
                absolute_tick,
                mid.ticks_per_beat,
                tempo_events
            )

            notes.append({
                "note": msg.note,
                "time": time_sec
            })

    return notes


# =========================
# ノートごと反転
# =========================

def make_note_restart_video(
    input_path,
    notes,
    output_path,
    progress_callback=None
):

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("動画を開けませんでした")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    source_frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    source_duration = (
        source_frame_count / fps
    )

    # 最初のノートを0秒にする
    first_midi_time = notes[0]["time"]

    note_times = [
        max(
            0.0,
            note["time"] - first_midi_time
        )
        for note in notes
    ]

    note_times = np.maximum.accumulate(
        note_times
    )

    # 最後のノートから動画1本分
    total_duration = (
        note_times[-1]
        + source_duration
    )

    output_frame_count = int(
        np.ceil(total_duration * fps)
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        cap.release()
        raise RuntimeError(
            "出力動画を作成できませんでした"
        )

    for output_frame in range(
        output_frame_count
    ):

        current_time = (
            output_frame / fps
        )

        segment = (
            np.searchsorted(
                note_times,
                current_time,
                side="right"
            ) - 1
        )

        if segment < 0:
            segment = 0

        segment_start = note_times[segment]

        local_time = (
            current_time - segment_start
        )

        source_frame = int(
            local_time * fps
        )

        source_frame = min(
            source_frame,
            source_frame_count - 1
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            source_frame
        )

        ret, frame = cap.read()

        if not ret:
            break

        # 奇数番目の区間だけ左右反転
        if segment % 2 == 1:
            frame = cv2.flip(frame, 1)

        out.write(frame)

        if progress_callback:
            progress_callback(
                (output_frame + 1)
                / output_frame_count
            )

    cap.release()
    out.release()


# =========================
# 動画そのまま反転
# =========================

def make_normal_video(
    input_path,
    notes,
    output_path,
    progress_callback=None
):

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("動画を開けませんでした")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    source_frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    source_duration = (
        source_frame_count / fps
    )

    # MIDIの最初のノートを0秒にする
    first_midi_time = notes[0]["time"]

    note_times = [
        max(
            0.0,
            note["time"] - first_midi_time
        )
        for note in notes
    ]

    note_times = np.maximum.accumulate(
        note_times
    )

    # MIDIの最後まで
    total_duration = max(
        source_duration,
        note_times[-1]
    )

    output_frame_count = int(
        np.ceil(total_duration * fps)
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        cap.release()
        raise RuntimeError(
            "出力動画を作成できませんでした"
        )

    last_frame = None

    note_index = 0

    for output_frame in range(
        output_frame_count
    ):

        current_time = (
            output_frame / fps
        )

        # 現在のノート位置
        while (
            note_index + 1 < len(note_times)
            and current_time >=
            note_times[note_index + 1]
        ):
            note_index += 1

        # 動画が残っている間は普通に読む
        if output_frame < source_frame_count:

            ret, frame = cap.read()

            if ret:
                last_frame = frame

        # 動画終了後は最後のフレーム
        if last_frame is None:
            continue

        frame = last_frame.copy()

        # 奇数番目のノートで左右反転
        if note_index % 2 == 1:
            frame = cv2.flip(frame, 1)

        out.write(frame)

        if progress_callback:
            progress_callback(
                (output_frame + 1)
                / output_frame_count
            )

    cap.release()
    out.release()


# =========================
# 上下左右反転
# =========================

def make_four_direction_video(
    input_path,
    notes,
    output_path,
    progress_callback=None
):

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("動画を開けませんでした")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    source_frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    source_duration = (
        source_frame_count / fps
    )

    # MIDIの最初のノートを0秒にする
    first_midi_time = notes[0]["time"]

    note_times = [
        max(
            0.0,
            note["time"] - first_midi_time
        )
        for note in notes
    ]

    note_times = np.maximum.accumulate(
        note_times
    )

    # MIDIの最後まで
    total_duration = max(
        source_duration,
        note_times[-1]
    )

    output_frame_count = int(
        np.ceil(total_duration * fps)
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        cap.release()
        raise RuntimeError(
            "出力動画を作成できませんでした"
        )

    last_frame = None

    note_index = 0

    for output_frame in range(
        output_frame_count
    ):

        current_time = (
            output_frame / fps
        )

        # ノート位置を更新
        while (
            note_index + 1 < len(note_times)
            and current_time >=
            note_times[note_index + 1]
        ):
            note_index += 1

        # 元動画を一度だけ再生
        if output_frame < source_frame_count:

            ret, frame = cap.read()

            if ret:
                last_frame = frame

        # 動画終了後は最後のフレーム
        if last_frame is None:
            continue

        frame = last_frame.copy()

        # 4種類を繰り返す
        mode = note_index % 4

        if mode == 0:
            # 通常
            pass

        elif mode == 1:
            # 左右反転
            frame = cv2.flip(frame, 1)

        elif mode == 2:
            # 上下反転
            frame = cv2.flip(frame, 0)

        elif mode == 3:
            # 上下左右反転
            frame = cv2.flip(frame, -1)

        out.write(frame)

        if progress_callback:
            progress_callback(
                (output_frame + 1)
                / output_frame_count
            )

    cap.release()
    out.release()


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="YTPMV向けMIDI反転ツール"
)

st.title("YTPMV向けMIDI反転ツール")


# =========================
# MIDI
# =========================

midi_file = st.file_uploader(
    "① MIDIファイルを選択",
    type=["mid", "midi"]
)

notes = None
track_index = None

if midi_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mid"
    ) as tmp:
        tmp.write(midi_file.read())
        midi_path = tmp.name

    mid = load_midi(midi_path)

    track_names = []

    for i, track in enumerate(mid.tracks):

        name = None

        for msg in track:

            if msg.type == "track_name":
                name = msg.name
                break

        if name:
            track_names.append(
                f"{i}: {name}"
            )
        else:
            track_names.append(
                f"{i}: Track {i}"
            )

    track_index = st.selectbox(
        "MIDIトラックを選択",
        range(len(track_names)),
        format_func=lambda i:
            track_names[i]
    )

    notes = get_notes_in_seconds(
        mid,
        track_index
    )


# =========================
# 動画
# =========================

video_file = st.file_uploader(
    "② 動画ファイルを選択",
    type=["mp4", "mov"]
)


# =========================
# 生成
# =========================

st.subheader("③ 生成")

mode = st.radio(
    "反転方式",
    [
        "ノートごと反転",
        "動画そのまま反転",
        "上下左右反転"
    ]
)


if midi_file and video_file and notes:

    if st.button(
        "🎬 動画を生成",
        use_container_width=True
    ):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as tmp_video:
            tmp_video.write(
                video_file.read()
            )
            input_video_path = (
                tmp_video.name
            )

        output_path = os.path.join(
            tempfile.gettempdir(),
            "ytpmv_output.mp4"
        )

        progress_bar = st.progress(0)

        percent_text = st.empty()

        def update_progress(value):

            percent = int(
                value * 100
            )

            progress_bar.progress(
                min(percent, 100)
            )

            percent_text.write(
                f"生成中… {percent}%"
            )

        try:

            if mode == "ノートごと反転":

                make_note_restart_video(
                    input_video_path,
                    notes,
                    output_path,
                    update_progress
                )

            elif mode == "動画そのまま反転":

                make_normal_video(
                    input_video_path,
                    notes,
                    output_path,
                    update_progress
                )

            elif mode == "上下左右反転":

                make_four_direction_video(
                    input_video_path,
                    notes,
                    output_path,
                    update_progress
                )

            progress_bar.progress(100)
            percent_text.write(
                "生成完了！ 100%"
            )

            st.video(output_path)

            with open(
                output_path,
                "rb"
            ) as f:

                st.download_button(
                    "⬇️ 動画をダウンロード",
                    f,
                    file_name="ytpmv_output.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"生成中にエラーが発生しました：{e}"
            )

        finally:

            if os.path.exists(
                input_video_path
            ):
                os.remove(
                    input_video_path
                )

else:

    st.info(
        "MIDIと動画を選択してください。"
    )
