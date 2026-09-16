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
    page_title="MIDI × Video YTPMV",
    layout="centered"
)

st.title("🎵 MIDI × 🎬 Video")
st.write("MIDIのノートに合わせて動画を再生・反転します。")


# =========================================================
# MIDI関連
# =========================================================

def get_track_name(track, index):
    """
    トラック名を安全に取得する。
    bytes や空文字になっている場合も処理する。
    """

    name = None

    for msg in track:
        if msg.is_meta and msg.type == "track_name":
            name = msg.name
            break

    if name is None:
        return f"Track {index}"

    # bytes の場合
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

    # 通常の文字列
    name = str(name).strip()

    if not name:
        return f"Track {index}"

    # "b''" のような表示になっている場合
    if name in ("b''", 'b""'):
        return f"Track {index}"

    return name


def get_note_on_events(track):
    """
    実際のノート開始イベントだけを取得する。

    note_on + velocity > 0
    をノート開始として扱う。

    note_on velocity 0 は note_off と同等なので
    音数には含めない。
    """

    events = []

    for msg in track:
        if msg.type == "note_on":
            velocity = getattr(msg, "velocity", 0)

            if velocity > 0:
                events.append(msg)

    return events


def count_notes(track):
    """
    トラックの音数を数える。
    """

    return len(get_note_on_events(track))


def get_tempo_events(mid):
    """
    MIDI全体からテンポ変更イベントを取得する。

    戻り値:
        [(absolute_tick, tempo), ...]
    """

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

    # 同じtickに複数テンポがある場合は最後のものを使用
    cleaned = []

    for tick, tempo in tempo_events:
        if cleaned and cleaned[-1][0] == tick:
            cleaned[-1] = (tick, tempo)
        else:
            cleaned.append((tick, tempo))

    # テンポ指定がない場合は120 BPM
    if not cleaned or cleaned[0][0] > 0:
        cleaned.insert(0, (0, mido.bpm2tempo(120)))

    return cleaned


def tick_to_seconds(tick, ticks_per_beat, tempo_events):
    """
    絶対tickを秒へ変換する。

    MIDIの途中でテンポが変わっても対応する。
    """

    if tick <= 0:
        return 0.0

    seconds = 0.0

    previous_tick = 0
    current_tempo = tempo_events[0][1]

    for tempo_tick, tempo in tempo_events:

        if tempo_tick > tick:
            break

        if tempo_tick > previous_tick:
            delta_ticks = tempo_tick - previous_tick

            seconds += (
                delta_ticks
                * current_tempo
                / 1_000_000.0
                / ticks_per_beat
            )

        previous_tick = tempo_tick
        current_tempo = tempo

    # 最後のテンポから指定tickまで
    if tick > previous_tick:
        delta_ticks = tick - previous_tick

        seconds += (
            delta_ticks
            * current_tempo
            / 1_000_000.0
            / ticks_per_beat
        )

    return seconds


def get_note_times(mid, track_index):
    """
    指定トラックのノート開始時刻を秒で取得。

    最初のノートを0秒として扱う。
    """

    track = mid.tracks[track_index]

    tempo_events = get_tempo_events(mid)

    absolute_tick = 0
    note_times = []

    for msg in track:
        absolute_tick += msg.time

        if msg.type == "note_on":
            velocity = getattr(msg, "velocity", 0)

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

    # 同じタイミングのコードは1回の切り替えとして扱う
    unique_times = []

    for t in note_times:
        if not unique_times:
            unique_times.append(t)
        elif abs(t - unique_times[-1]) > 0.000001:
            unique_times.append(t)

    return unique_times


# =========================================================
# 動画関連
# =========================================================

def open_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("動画を開けませんでした。")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration = frame_count / fps if fps > 0 else 0

    return cap, fps, width, height, frame_count, duration


def create_writer(output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("出力動画を作成できませんでした。")

    return writer


def flip_frame(frame, mode):
    """
    mode:
        0 = 通常
        1 = 左右反転
        2 = 上下反転
        3 = 上下左右反転
    """

    if mode == 0:
        return frame

    if mode == 1:
        return cv2.flip(frame, 1)

    if mode == 2:
        return cv2.flip(frame, 0)

    if mode == 3:
        return cv2.flip(frame, -1)

    return frame


def read_source_frames(video_path):
    """
    動画をメモリに読み込む。

    小～中程度の動画を想定。
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("動画を開けませんでした。")

    fps = cap.get(cv2.CAP_PROP_FPS)

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
        raise RuntimeError("動画からフレームを読み込めませんでした。")

    height, width = frames[0].shape[:2]

    return frames, fps, width, height


# =========================================================
# モード1
# 反転なし（ノートごと再生）
# =========================================================

def make_no_flip_video(
    video_path,
    note_times,
    output_path,
    progress_callback=None
):
    frames, fps, width, height = read_source_frames(video_path)

    writer = create_writer(
        output_path,
        fps,
        width,
        height
    )

    source_frame_count = len(frames)

    # ノートがない場合
    if not note_times:
        for i, frame in enumerate(frames):
            writer.write(frame)

            if progress_callback:
                progress_callback(
                    (i + 1) / source_frame_count
                )

        writer.release()
        return

    # 各ノートから次のノートまで
    for note_index in range(len(note_times)):

        start_time = note_times[note_index]

        if note_index + 1 < len(note_times):
            end_time = note_times[note_index + 1]
        else:
            # 最後のノート以降は動画を最後まで再生
            end_time = start_time + (
                source_frame_count / fps
            )

        duration = max(0.0, end_time - start_time)

        frame_count = max(
            1,
            int(round(duration * fps))
        )

        for i in range(frame_count):

            source_index = i

            if source_index >= source_frame_count:
                break

            writer.write(frames[source_index])

        if progress_callback:
            progress_callback(
                (note_index + 1) / len(note_times)
            )

    writer.release()


# =========================================================
# モード2
# ノートごと反転
# =========================================================

def make_note_restart_video(
    video_path,
    note_times,
    output_path,
    progress_callback=None
):
    frames, fps, width, height = read_source_frames(video_path)

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

    for note_index in range(len(note_times)):

        if note_index + 1 < len(note_times):
            duration = (
                note_times[note_index + 1]
                - note_times[note_index]
            )
        else:
            duration = source_frame_count / fps

        duration = max(0.0, duration)

        frame_count = max(
            1,
            int(round(duration * fps))
        )

        # 奇数ノートだけ左右反転
        flip_mode = note_index % 2

        for i in range(frame_count):

            source_index = i

            if source_index >= source_frame_count:
                break

            frame = frames[source_index]

            frame = flip_frame(
                frame,
                flip_mode
            )

            writer.write(frame)

        if progress_callback:
            progress_callback(
                (note_index + 1) / len(note_times)
            )

    writer.release()


# =========================================================
# モード3
# 動画そのまま反転
# =========================================================

def make_normal_video(
    video_path,
    note_times,
    output_path,
    progress_callback=None
):
    frames, fps, width, height = read_source_frames(video_path)

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
                    (i + 1) / source_frame_count
                )

        writer.release()
        return

    last_note_time = note_times[-1]

    video_duration = source_frame_count / fps

    # 最後のノートまで最低限出力
    output_duration = max(
        video_duration,
        last_note_time + 1.0 / fps
    )

    output_frame_count = int(
        np.ceil(output_duration * fps)
    )

    note_index = 0

    for frame_index in range(output_frame_count):

        current_time = frame_index / fps

        # 現在時刻までに何回ノートが来たか
        while (
            note_index + 1 < len(note_times)
            and current_time >= note_times[note_index + 1]
        ):
            note_index += 1

        # ノート番号によって反転
        flip_mode = note_index % 2

        # 動画が残っている場合
        if frame_index < source_frame_count:
            frame = frames[frame_index]

        else:
            # 動画終了後は最後のフレームを保持
            frame = frames[-1]

        frame = flip_frame(
            frame,
            flip_mode
        )

        writer.write(frame)

        if progress_callback:
            progress_callback(
                (frame_index + 1) / output_frame_count
            )

    writer.release()


# =========================================================
# モード4
# 上下左右反転
# =========================================================

def make_four_direction_video(
    video_path,
    note_times,
    output_path,
    progress_callback=None
):
    frames, fps, width, height = read_source_frames(video_path)

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
                    (i + 1) / source_frame_count
                )

        writer.release()
        return

    # 通常 → 左右 → 上下 → 上下左右
    flip_modes = [0, 1, 2, 3]

    for note_index in range(len(note_times)):

        if note_index + 1 < len(note_times):
            duration = (
                note_times[note_index + 1]
                - note_times[note_index]
            )
        else:
            duration = source_frame_count / fps

        duration = max(0.0, duration)

        frame_count = max(
            1,
            int(round(duration * fps))
        )

        flip_mode = flip_modes[
            note_index % 4
        ]

        for i in range(frame_count):

            source_index = i

            if source_index >= source_frame_count:
                break

            frame = frames[source_index]

            frame = flip_frame(
                frame,
                flip_mode
            )

            writer.write(frame)

        if progress_callback:
            progress_callback(
                (note_index + 1) / len(note_times)
            )

    writer.release()


# =========================================================
# MIDIアップロード
# =========================================================

midi_file = st.file_uploader(
    "🎵 MIDIファイル",
    type=["mid", "midi"]
)


# =========================================================
# 動画アップロード
# =========================================================

video_file = st.file_uploader(
    "🎬 動画ファイル",
    type=["mp4", "mov"]
)


# =========================================================
# MIDI解析
# =========================================================

mid = None
midi_temp_path = None

if midi_file is not None:

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mid"
        ) as tmp:
            tmp.write(midi_file.getbuffer())
            midi_temp_path = tmp.name

        mid = mido.MidiFile(midi_temp_path)

        st.success("MIDIを読み込みました。")

        # -------------------------------------------------
        # トラック解析
        # -------------------------------------------------

        track_options = []

        for index, track in enumerate(mid.tracks):

            note_count = count_notes(track)

            # 音数0は表示しない
            if note_count <= 0:
                continue

            track_name = get_track_name(
                track,
                index
            )

            label = (
                f"{index}: "
                f"{track_name} "
                f"（音数: {note_count}）"
            )

            track_options.append(
                (index, label, note_count)
            )

        # -------------------------------------------------
        # トラック選択
        # -------------------------------------------------

        if track_options:

            st.subheader("🎹 MIDIトラックを選択")

            labels = [
                item[1]
                for item in track_options
            ]

            selected_label = st.selectbox(
                "音数0のトラックは表示されません",
                labels
            )

            selected_track_index = next(
                item[0]
                for item in track_options
                if item[1] == selected_label
            )

            selected_note_count = next(
                item[2]
                for item in track_options
                if item[0] == selected_track_index
            )

            st.info(
                f"選択中："
                f"トラック {selected_track_index} / "
                f"音数 {selected_note_count}"
            )

        else:

            st.error(
                "ノートが入っているMIDIトラックが見つかりませんでした。"
            )

            selected_track_index = None

    except Exception as e:

        st.error(
            f"MIDIの読み込みに失敗しました：{e}"
        )

        selected_track_index = None


# =========================================================
# モード選択
# =========================================================

mode = st.radio(
    "🎬 再生モード",
    [
        "反転なし（ノートごと再生）",
        "ノートごと反転",
        "動画そのまま反転",
        "上下左右反転"
    ]
)


# =========================================================
# 動画プレビュー
# =========================================================

if video_file is not None:

    st.video(video_file)


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

            # ---------------------------------------------
            # 動画を一時保存
            # ---------------------------------------------

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

            # ---------------------------------------------
            # MIDIノート時刻取得
            # ---------------------------------------------

            note_times = get_note_times(
                mid,
                selected_track_index
            )

            if not note_times:

                st.error(
                    "選択したトラックにノートがありません。"
                )

                st.stop()

            st.write(
                f"🎵 ノート数：{len(note_times)}"
            )

            st.write(
                f"⏱ 最初のノート：0.000秒"
            )

            st.write(
                f"⏱ 最後のノート："
                f"{note_times[-1]:.3f}秒"
            )

            # ---------------------------------------------
            # 出力先
            # ---------------------------------------------

            output_path = os.path.join(
                tempfile.gettempdir(),
                "ytpmv_output.mp4"
            )

            progress_bar = st.progress(0)

            def update_progress(value):
                progress_bar.progress(
                    min(1.0, max(0.0, value))
                )

            # ---------------------------------------------
            # モード別生成
            # ---------------------------------------------

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

            progress_bar.progress(1.0)

            # ---------------------------------------------
            # 完成
            # ---------------------------------------------

            st.success(
                "🎉 動画が完成しました！"
            )

            st.video(output_path)

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
                    os.remove(video_temp_path)
                except Exception:
                    pass

            if midi_temp_path:
                try:
                    os.remove(midi_temp_path)
                except Exception:
                    pass
