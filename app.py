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
st.write("スマホ勢向けの動画反転ツールです。パソコン勢の方もどうぞ.")


# =========================================================
# MIDI
# =========================================================

def get_track_name(track, index):
    """MIDIトラック名を取得"""

    for msg in track:
        if msg.type == "track_name":

            name = msg.name

            if isinstance(name, bytes):
                try:
                    name = name.decode(
                        "utf-8",
                        errors="ignore"
                    )
                except Exception:
                    name = ""

            if name and str(name).strip():
                return str(name).strip()

    return f"トラック{index}"


def get_note_on_events(track):
    """
    velocity > 0 の note_on を取得。

    tickはそのトラック内での絶対tick。
    """

    events = []

    absolute_tick = 0

    for msg in track:

        absolute_tick += msg.time

        if (
            msg.type == "note_on"
            and getattr(msg, "velocity", 0) > 0
        ):

            events.append({
                "tick": absolute_tick,
                "note": getattr(msg, "note", 0)
            })

    return events


def count_notes(track):
    """ノート数"""

    return len(
        get_note_on_events(track)
    )


# =========================================================
# MIDI全体のテンポマップ
# =========================================================

def build_tempo_map(mid):
    """
    MIDI全体からテンポマップを作る。

    重要：
    テンポは選択したトラックだけを見るのではなく、
    MIDI全トラックを統合したグローバル時間軸から取得する。

    MIDIのテンポ変更が途中にあっても対応する。
    """

    # 全トラックをMIDIのグローバル時間軸に統合
    merged = mido.merge_tracks(
        mid.tracks
    )

    tempo_events = []

    absolute_tick = 0

    for msg in merged:

        absolute_tick += msg.time

        if msg.type == "set_tempo":

            tempo_events.append(
                (
                    absolute_tick,
                    msg.tempo
                )
            )

    # テンポイベントがない場合
    # MIDI標準のデフォルト 120 BPM
    if not tempo_events:

        return [
            (
                0,
                mido.bpm2tempo(120)
            )
        ]

    # tick順
    tempo_events.sort(
        key=lambda x: x[0]
    )

    # 同じtickのテンポ変更を整理
    merged_tempo = []

    for tick, tempo in tempo_events:

        if (
            merged_tempo
            and merged_tempo[-1][0] == tick
        ):

            # 同じtickなら後のものを採用
            merged_tempo[-1] = (
                tick,
                tempo
            )

        else:

            merged_tempo.append(
                (
                    tick,
                    tempo
                )
            )

    # tick=0にテンポがない場合
    if merged_tempo[0][0] != 0:

        merged_tempo.insert(
            0,
            (
                0,
                mido.bpm2tempo(120)
            )
        )

    return merged_tempo


def tick_to_seconds(
    tick,
    ticks_per_beat,
    tempo_map
):
    """
    絶対tick → 秒

    MIDI全体のテンポマップを使用する。
    """

    if tick <= 0:
        return 0.0

    seconds = 0.0

    previous_tick = 0

    current_tempo = tempo_map[0][1]

    for event_tick, event_tempo in tempo_map:

        if event_tick > tick:
            break

        delta_ticks = (
            event_tick
            - previous_tick
        )

        if delta_ticks > 0:

            seconds += (
                delta_ticks
                * current_tempo
                / 1_000_000.0
                / ticks_per_beat
            )

        previous_tick = event_tick

        current_tempo = event_tempo

    remaining_ticks = (
        tick
        - previous_tick
    )

    if remaining_ticks > 0:

        seconds += (
            remaining_ticks
            * current_tempo
            / 1_000_000.0
            / ticks_per_beat
        )

    return seconds


def get_note_times(mid, track):
    """
    選択トラックのノート時刻を秒に変換。

    MIDI全体のテンポマップを使用する。
    """

    note_events = get_note_on_events(
        track
    )

    if not note_events:
        return []

    # MIDI全体のテンポを取得
    tempo_map = build_tempo_map(
        mid
    )

    raw_times = []

    for event in note_events:

        seconds = tick_to_seconds(
            event["tick"],
            mid.ticks_per_beat,
            tempo_map
        )

        raw_times.append(
            seconds
        )

    if not raw_times:
        return []

    # 最初のノートを0秒にする
    first_time = raw_times[0]

    note_times = [
        max(
            0.0,
            t - first_time
        )
        for t in raw_times
    ]

    # 同時発音は同じタイミングとしてまとめる
    unique_times = []

    for t in note_times:

        if not unique_times:

            unique_times.append(t)

        elif abs(
            t - unique_times[-1]
        ) > 1e-9:

            unique_times.append(t)

    return unique_times


# =========================================================
# 動画
# =========================================================

def flip_frame(frame, mode):
    """
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


def create_writer(
    output_path,
    width,
    height,
    fps
):

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    return cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (
            width,
            height
        )
    )


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

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    if not frames:

        raise RuntimeError(
            "動画のフレームを読み込めませんでした。"
        )

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
    frames,
    fps,
    note_times,
    output_path,
    update_progress
):

    height, width = frames[0].shape[:2]

    writer = create_writer(
        output_path,
        width,
        height,
        fps
    )

    source_count = len(frames)

    if not note_times:

        for i, frame in enumerate(
            frames
        ):

            writer.write(frame)

            update_progress(
                (i + 1)
                / source_count
            )

        writer.release()
        return

    total_segments = len(
        note_times
    )

    for segment_index in range(
        total_segments
    ):

        # -------------------------------------------------
        # MIDI上の正確な区間時間
        # -------------------------------------------------

        if (
            segment_index + 1
            < total_segments
        ):

            duration = (
                note_times[
                    segment_index + 1
                ]
                - note_times[
                    segment_index
                ]
            )

            frame_count = max(
                1,
                int(
                    round(
                        duration * fps
                    )
                )
            )

        else:

            # 最後のノート後は
            # 元動画を最後まで再生
            frame_count = source_count

        # -------------------------------------------------
        # 動画を0秒から再生
        # -------------------------------------------------

        for frame_index in range(
            frame_count
        ):

            source_index = min(
                frame_index,
                source_count - 1
            )

            writer.write(
                frames[source_index]
            )

        update_progress(
            (segment_index + 1)
            / total_segments
        )

    writer.release()


# =========================================================
# ② ノートごと反転
# =========================================================

def make_note_restart_video(
    frames,
    fps,
    note_times,
    output_path,
    update_progress
):

    height, width = frames[0].shape[:2]

    writer = create_writer(
        output_path,
        width,
        height,
        fps
    )

    source_count = len(frames)

    if not note_times:

        for i, frame in enumerate(
            frames
        ):

            writer.write(frame)

            update_progress(
                (i + 1)
                / source_count
            )

        writer.release()
        return

    total_segments = len(
        note_times
    )

    for segment_index in range(
        total_segments
    ):

        # -------------------------------------------------
        # ノート間の正確な時間
        # -------------------------------------------------

        if (
            segment_index + 1
            < total_segments
        ):

            duration = (
                note_times[
                    segment_index + 1
                ]
                - note_times[
                    segment_index
                ]
            )

            frame_count = max(
                1,
                int(
                    round(
                        duration * fps
                    )
                )
            )

        else:

            frame_count = source_count

        # -------------------------------------------------
        # 通常 → 左右反転 → 通常...
        # -------------------------------------------------

        flip_mode = (
            segment_index % 2
        )

        for frame_index in range(
            frame_count
        ):

            source_index = min(
                frame_index,
                source_count - 1
            )

            frame = frames[
                source_index
            ]

            frame = flip_frame(
                frame,
                flip_mode
            )

            writer.write(frame)

        update_progress(
            (segment_index + 1)
            / total_segments
        )

    writer.release()


# =========================================================
# ③ 動画そのまま反転
# =========================================================

def make_normal_video(
    frames,
    fps,
    note_times,
    output_path,
    update_progress
):

    height, width = frames[0].shape[:2]

    source_count = len(frames)

    source_duration = (
        source_count
        / fps
    )

    midi_duration = 0.0

    if note_times:

        midi_duration = (
            note_times[-1]
        )

    output_duration = max(
        source_duration,
        midi_duration
    )

    total_frames = max(
        1,
        int(
            round(
                output_duration
                * fps
            )
        )
    )

    writer = create_writer(
        output_path,
        width,
        height,
        fps
    )

    # ノート時刻をフレーム番号へ
    note_frame_positions = [
        int(
            round(
                t * fps
            )
        )
        for t in note_times
    ]

    for output_index in range(
        total_frames
    ):

        # 元動画は一度だけ再生
        source_index = min(
            output_index,
            source_count - 1
        )

        frame = frames[
            source_index
        ]

        # ノートが何回通過したか
        transition_count = (
            np.searchsorted(
                note_frame_positions,
                output_index,
                side="right"
            )
        )

        # 通常 → 左右反転 → 通常...
        flip_mode = (
            transition_count % 2
        )

        frame = flip_frame(
            frame,
            flip_mode
        )

        writer.write(frame)

        if (
            output_index
            % max(
                1,
                total_frames // 100
            )
            == 0
        ):

            update_progress(
                (output_index + 1)
                / total_frames
            )

    writer.release()

    update_progress(1.0)


# =========================================================
# ④ 上下左右反転
# =========================================================

def make_four_direction_video(
    frames,
    fps,
    note_times,
    output_path,
    update_progress
):

    height, width = frames[0].shape[:2]

    writer = create_writer(
        output_path,
        width,
        height,
        fps
    )

    source_count = len(frames)

    if not note_times:

        for i, frame in enumerate(
            frames
        ):

            writer.write(frame)

            update_progress(
                (i + 1)
                / source_count
            )

        writer.release()
        return

    total_segments = len(
        note_times
    )

    for segment_index in range(
        total_segments
    ):

        if (
            segment_index + 1
            < total_segments
        ):

            duration = (
                note_times[
                    segment_index + 1
                ]
                - note_times[
                    segment_index
                ]
            )

            frame_count = max(
                1,
                int(
                    round(
                        duration * fps
                    )
                )
            )

        else:

            frame_count = source_count

        # ---------------------------------------------
        # 通常
        # ↓
        # 左右
        # ↓
        # 上下
        # ↓
        # 上下左右
        # ↓
        # 通常...
        # ---------------------------------------------

        flip_mode = (
            segment_index % 4
        )

        for frame_index in range(
            frame_count
        ):

            source_index = min(
                frame_index,
                source_count - 1
            )

            frame = frames[
                source_index
            ]

            frame = flip_frame(
                frame,
                flip_mode
            )

            writer.write(frame)

        update_progress(
            (segment_index + 1)
            / total_segments
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

mid = None
selected_track = None


if midi_file is not None:

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mid"
        ) as tmp:

            tmp.write(
                midi_file.getbuffer()
            )

            midi_path = tmp.name

        mid = mido.MidiFile(
            midi_path
        )

        os.unlink(
            midi_path
        )

        # =================================================
        # ② トラック
        # =================================================

        st.subheader(
            "②トラック"
        )

        track_options = []

        for index, track in enumerate(
            mid.tracks
        ):

            note_count = count_notes(
                track
            )

            # ノート0のトラックは表示しない
            if note_count > 0:

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
                        track
                    )
                )

        if not track_options:

            st.error(
                "ノートが入っているトラックがありません。"
            )

        else:

            labels = [
                item[1]
                for item in track_options
            ]

            selected_label = st.selectbox(
                "使用するトラック",
                labels
            )

            selected_item = next(
                item
                for item in track_options
                if item[1] == selected_label
            )

            selected_track = (
                selected_item[2]
            )

    except Exception as e:

        st.error(
            f"MIDIを読み込めませんでした: {e}"
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
# ④ 反転
# =========================================================

st.subheader(
    "④反転"
)

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
# 生成
# =========================================================

if st.button(
    "🎬 動画を生成",
    type="primary"
):

    # -----------------------------------------------------
    # チェック
    # -----------------------------------------------------

    if mid is None:

        st.error(
            "先にMIDIを選択してください。"
        )

        st.stop()

    if selected_track is None:

        st.error(
            "トラックを選択してください。"
        )

        st.stop()

    if video_file is None:

        st.error(
            "先に動画を選択してください。"
        )

        st.stop()

    # -----------------------------------------------------
    # MIDIノート時刻
    # -----------------------------------------------------

    try:

        note_times = get_note_times(
            mid,
            selected_track
        )

    except Exception as e:

        st.error(
            f"MIDIの解析に失敗しました: {e}"
        )

        st.stop()

    if not note_times:

        st.error(
            "選択したトラックにノートがありません。"
        )

        st.stop()

    # -----------------------------------------------------
    # 動画保存
    # -----------------------------------------------------

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as tmp:

            tmp.write(
                video_file.getbuffer()
            )

            video_path = tmp.name

    except Exception as e:

        st.error(
            f"動画の保存に失敗しました: {e}"
        )

        st.stop()

    # -----------------------------------------------------
    # 出力
    # -----------------------------------------------------

    output_path = os.path.join(
        tempfile.gettempdir(),
        "ytpmv_output.mp4"
    )

    # -----------------------------------------------------
    # ％表示
    # -----------------------------------------------------

    progress_text = st.empty()

    progress_bar = st.progress(
        0
    )

    def update_progress(value):

        value = min(
            1.0,
            max(
                0.0,
                value
            )
        )

        percent = int(
            value * 100
        )

        progress_text.write(
            f"動画を生成中… {percent}%"
        )

        progress_bar.progress(
            value
        )

    update_progress(0)

    # -----------------------------------------------------
    # 元動画読み込み
    # -----------------------------------------------------

    try:

        (
            frames,
            fps,
            width,
            height
        ) = read_source_frames(
            video_path
        )

    except Exception as e:

        progress_text.empty()
        progress_bar.empty()

        st.error(
            f"動画を読み込めませんでした: {e}"
        )

        try:
            os.unlink(
                video_path
            )
        except Exception:
            pass

        st.stop()

    # -----------------------------------------------------
    # 生成
    # -----------------------------------------------------

    try:

        if mode == (
            "反転なし（ノートごと再生）"
        ):

            make_no_flip_video(
                frames,
                fps,
                note_times,
                output_path,
                update_progress
            )

        elif mode == "ノートごと反転":

            make_note_restart_video(
                frames,
                fps,
                note_times,
                output_path,
                update_progress
            )

        elif mode == "動画そのまま反転":

            make_normal_video(
                frames,
                fps,
                note_times,
                output_path,
                update_progress
            )

        elif mode == "上下左右反転":

            make_four_direction_video(
                frames,
                fps,
                note_times,
                output_path,
                update_progress
            )

        # -------------------------------------------------
        # 完了
        # -------------------------------------------------

        progress_text.write(
            "動画を生成中… 100%"
        )

        progress_bar.progress(
            1.0
        )

        st.success(
            "動画の生成が完了しました！"
        )

        # -------------------------------------------------
        # ダウンロード
        # -------------------------------------------------

        with open(
            output_path,
            "rb"
        ) as f:

            video_bytes = f.read()

        st.download_button(
            label="⬇️ 動画を保存",
            data=video_bytes,
            file_name="ytpmv_output.mp4",
            mime="video/mp4"
        )

    except Exception as e:

        st.error(
            f"動画の生成に失敗しました: {e}"
        )

    finally:

        try:
            os.unlink(
                video_path
            )
        except Exception:
            pass
