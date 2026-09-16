import os
import tempfile

import streamlit as st
import mido
import cv2
import numpy as np


# =========================
# ページ設定
# =========================

st.set_page_config(
    page_title="YTPMV向けMIDI反転ツール",
    layout="centered"
)

st.title("YTPMV向けMIDI反転ツール")
st.write("スマホ勢向けの動画反転ツールです。パソコン勢の方もどうぞ.")


# =========================
# MIDI関連
# =========================

def get_track_name(track, index):
    """
    MIDIトラック名を取得する。
    トラック名がない場合は「トラックX」とする。
    """

    for msg in track:
        if msg.type == "track_name":
            name = msg.name

            if isinstance(name, bytes):
                try:
                    name = name.decode("utf-8", errors="ignore")
                except Exception:
                    name = ""

            if name and str(name).strip():
                return str(name).strip()

    return f"トラック{index}"


def get_note_on_events(track):
    """
    velocity > 0 の note_on を取得。
    """

    events = []

    absolute_tick = 0

    for msg in track:
        absolute_tick += msg.time

        if (
            msg.type == "note_on"
            and getattr(msg, "velocity", 0) > 0
        ):
            events.append(
                {
                    "tick": absolute_tick,
                    "note": getattr(msg, "note", 0)
                }
            )

    return events


def count_notes(track):
    """
    トラック内のノート数。
    """

    return len(get_note_on_events(track))


def get_tempo_events(mid):
    """
    MIDI全体からテンポ変更イベントを取得。

    tick:
        絶対tick

    tempo:
        マイクロ秒 / 4分音符
    """

    tempo_events = []

    for track in mid.tracks:
        absolute_tick = 0

        for msg in track:
            absolute_tick += msg.time

            if msg.type == "set_tempo":
                tempo_events.append(
                    (
                        absolute_tick,
                        msg.tempo
                    )
                )

    # 最初のテンポがない場合は120 BPM
    if not tempo_events:
        tempo_events = [(0, mido.bpm2tempo(120))]

    tempo_events.sort(key=lambda x: x[0])

    # 同じtickに複数ある場合は最後のものを使用
    merged = []

    for tick, tempo in tempo_events:
        if merged and merged[-1][0] == tick:
            merged[-1] = (tick, tempo)
        else:
            merged.append((tick, tempo))

    # tick 0 のテンポがない場合
    if merged[0][0] != 0:
        merged.insert(
            0,
            (0, mido.bpm2tempo(120))
        )

    return merged


def tick_to_seconds(
    tick,
    ticks_per_beat,
    tempo_events
):
    """
    絶対tickを秒へ変換。

    MIDI途中のテンポ変更にも対応。
    """

    seconds = 0.0

    previous_tick = 0
    current_tempo = tempo_events[0][1]

    for event_tick, event_tempo in tempo_events:

        if event_tick > tick:
            break

        delta_ticks = event_tick - previous_tick

        seconds += (
            delta_ticks
            * current_tempo
            / 1_000_000
            / ticks_per_beat
        )

        previous_tick = event_tick
        current_tempo = event_tempo

    remaining_ticks = tick - previous_tick

    seconds += (
        remaining_ticks
        * current_tempo
        / 1_000_000
        / ticks_per_beat
    )

    return seconds


def get_note_times(mid, track):
    """
    MIDIノートの発音位置を秒で取得。

    最初のノートが必ず0秒になるようにする。
    """

    note_events = get_note_on_events(track)

    if not note_events:
        return []

    tempo_events = get_tempo_events(mid)

    raw_times = []

    for event in note_events:

        seconds = tick_to_seconds(
            event["tick"],
            mid.ticks_per_beat,
            tempo_events
        )

        raw_times.append(seconds)

    # 最初のノートを0秒に合わせる
    first_time = raw_times[0]

    note_times = [
        max(0.0, t - first_time)
        for t in raw_times
    ]

    # 同じ時刻のノートは1つの境界として扱う
    unique_times = []

    for t in note_times:

        if not unique_times:
            unique_times.append(t)

        elif abs(t - unique_times[-1]) > 1e-9:
            unique_times.append(t)

    return unique_times


# =========================
# 動画関連
# =========================

def flip_frame(frame, mode):
    """
    フレームを反転する。

    mode:
        0 = 反転なし
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
    """
    MP4出力用VideoWriter。
    """

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    return writer


def read_source_frames(video_path):
    """
    元動画を全フレーム読み込み。

    メモリ上に保持しておくことで、
    ノートごとの再生を高速化する。
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            "動画を開けませんでした。"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
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

    return frames, fps, width, height


# =========================
# ① 反転なし
# =========================

def make_no_flip_video(
    frames,
    fps,
    note_times,
    output_path,
    update_progress
):
    """
    反転なし（ノートごと再生）

    各ノートで動画を0秒から再スタート。
    最後のノート後も動画を最後まで再生。
    """

    height, width = frames[0].shape[:2]

    writer = create_writer(
        output_path,
        width,
        height,
        fps
    )

    source_count = len(frames)

    # ノートがない場合
    if not note_times:

        for i, frame in enumerate(frames):

            writer.write(frame)

            update_progress(
                (i + 1) / source_count
            )

        writer.release()
        return

    total_segments = len(note_times)

    for segment_index in range(total_segments):

        # 各ノートから次のノートまで
        if segment_index + 1 < total_segments:

            duration = (
                note_times[segment_index + 1]
                - note_times[segment_index]
            )

            frame_count = max(
                1,
                int(round(duration * fps))
            )

        else:
            # 最後のノート後は元動画全体
            frame_count = source_count

        for frame_index in range(frame_count):

            source_index = min(
                frame_index,
                source_count - 1
            )

            writer.write(
                frames[source_index]
            )

        progress = (
            (segment_index + 1)
            / total_segments
        )

        update_progress(progress)

    writer.release()


# =========================
# ② ノートごと反転
# =========================

def make_note_restart_video(
    frames,
    fps,
    note_times,
    output_path,
    update_progress
):
    """
    ノートごと反転

    ノートごとに動画を0秒から再スタート。

    ノート1: 通常
    ノート2: 左右反転
    ノート3: 通常
    ノート4: 左右反転
    ...
    """

    height, width = frames[0].shape[:2]

    writer = create_writer(
        output_path,
        width,
        height,
        fps
    )

    source_count = len(frames)

    if not note_times:

        for i, frame in enumerate(frames):

            writer.write(frame)

            update_progress(
                (i + 1) / source_count
            )

        writer.release()
        return

    total_segments = len(note_times)

    for segment_index in range(total_segments):

        # ノート間の長さ
        if segment_index + 1 < total_segments:

            duration = (
                note_times[segment_index + 1]
                - note_times[segment_index]
            )

            frame_count = max(
                1,
                int(round(duration * fps))
            )

        else:
            # 最後のノート後は元動画全体
            frame_count = source_count

        # ノート番号
        # 0 = 通常
        # 1 = 左右反転
        flip_mode = (
            segment_index % 2
        )

        for frame_index in range(frame_count):

            source_index = min(
                frame_index,
                source_count - 1
            )

            frame = frames[source_index]

            frame = flip_frame(
                frame,
                flip_mode
            )

            writer.write(frame)

        progress = (
            (segment_index + 1)
            / total_segments
        )

        update_progress(progress)

    writer.release()


# =========================
# ③ 動画そのまま反転
# =========================

def make_normal_video(
    frames,
    fps,
    note_times,
    output_path,
    update_progress
):
    """
    動画そのまま反転

    元動画は最初から最後まで1回だけ再生。

    ノート1: 通常
    ノート2: 左右反転
    ノート3: 通常
    ノート4: 左右反転
    ...

    動画終了後も必要なら最後のフレームを保持。
    """

    height, width = frames[0].shape[:2]

    source_count = len(frames)

    source_duration = (
        source_count / fps
    )

    # MIDI側の最後のノートまで必要
    midi_duration = 0.0

    if note_times:
        midi_duration = note_times[-1]

    # 少なくとも元動画分は出力
    output_duration = max(
        source_duration,
        midi_duration
    )

    total_frames = max(
        1,
        int(
            round(
                output_duration * fps
            )
        )
    )

    writer = create_writer(
        output_path,
        width,
        height,
        fps
    )

    # ノート位置をフレーム番号に変換
    note_frame_positions = [
        int(round(t * fps))
        for t in note_times
    ]

    note_frame_positions = [
        max(0, x)
        for x in note_frame_positions
    ]

    for output_index in range(total_frames):

        # 元動画のフレーム
        source_index = min(
            output_index,
            source_count - 1
        )

        frame = frames[source_index]

        # 現在何回反転が切り替わったか
        transition_count = (
            np.searchsorted(
                note_frame_positions,
                output_index,
                side="right"
            )
        )

        # 0 = 通常
        # 1 = 左右反転
        flip_mode = (
            transition_count % 2
        )

        frame = flip_frame(
            frame,
            flip_mode
        )

        writer.write(frame)

        if output_index % max(
            1,
            total_frames // 100
        ) == 0:

            update_progress(
                (output_index + 1)
                / total_frames
            )

    writer.release()

    update_progress(1.0)


# =========================
# ④ 上下左右反転
# =========================

def make_four_direction_video(
    frames,
    fps,
    note_times,
    output_path,
    update_progress
):
    """
    上下左右反転

    各ノートで動画を0秒から再スタート。

    ノート1: 通常
    ノート2: 左右反転
    ノート3: 上下反転
    ノート4: 上下左右反転
    ノート5: 通常
    ...
    """

    height, width = frames[0].shape[:2]

    writer = create_writer(
        output_path,
        width,
        height,
        fps
    )

    source_count = len(frames)

    if not note_times:

        for i, frame in enumerate(frames):

            writer.write(frame)

            update_progress(
                (i + 1) / source_count
            )

        writer.release()
        return

    total_segments = len(note_times)

    for segment_index in range(total_segments):

        if segment_index + 1 < total_segments:

            duration = (
                note_times[segment_index + 1]
                - note_times[segment_index]
            )

            frame_count = max(
                1,
                int(round(duration * fps))
            )

        else:
            # 最後のノート後は元動画全体
            frame_count = source_count

        # 4方向を繰り返す
        flip_mode = (
            segment_index % 4
        )

        for frame_index in range(frame_count):

            source_index = min(
                frame_index,
                source_count - 1
            )

            frame = frames[source_index]

            frame = flip_frame(
                frame,
                flip_mode
            )

            writer.write(frame)

        progress = (
            (segment_index + 1)
            / total_segments
        )

        update_progress(progress)

    writer.release()


# =========================
# ① MIDI
# =========================

midi_file = st.file_uploader(
    "①MIDI",
    type=["mid", "midi"]
)

mid = None
selected_track = None
note_times = []


if midi_file is not None:

    try:

        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mid"
        ) as tmp:

            tmp.write(
                midi_file.getbuffer()
            )

            midi_path = tmp.name

        # MIDI読み込み
        mid = mido.MidiFile(
            midi_path
        )

        os.unlink(midi_path)

        # =========================
        # ② トラック
        # =========================

        st.subheader("②トラック")

        track_options = []

        for index, track in enumerate(
            mid.tracks
        ):

            note_count = count_notes(
                track
            )

            # 音数0のトラックは表示しない
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


# =========================
# ③ 動画
# =========================

video_file = st.file_uploader(
    "③動画",
    type=["mp4", "mov"]
)


# =========================
# ④ 反転
# =========================

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


# =========================
# 動画生成
# =========================

if st.button(
    "🎬 動画を生成",
    type="primary"
):

    # MIDIチェック
    if mid is None:

        st.error(
            "先にMIDIを選択してください。"
        )

        st.stop()

    # トラックチェック
    if selected_track is None:

        st.error(
            "トラックを選択してください。"
        )

        st.stop()

    # 動画チェック
    if video_file is None:

        st.error(
            "先に動画を選択してください。"
        )

        st.stop()

    # =========================
    # MIDIノート時刻取得
    # =========================

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

    # =========================
    # 動画を一時保存
    # =========================

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

    # =========================
    # 出力先
    # =========================

    output_path = os.path.join(
        tempfile.gettempdir(),
        "ytpmv_output.mp4"
    )

    # =========================
    # ％表示
    # =========================

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

    # 開始表示
    update_progress(0)

    # =========================
    # 動画読み込み
    # =========================

    try:

        frames, fps, width, height = (
            read_source_frames(
                video_path
            )
        )

    except Exception as e:

        progress_text.empty()
        progress_bar.empty()

        st.error(
            f"動画を読み込めませんでした: {e}"
        )

        try:
            os.unlink(video_path)
        except Exception:
            pass

        st.stop()

    # =========================
    # 動画生成
    # =========================

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

        # 完了
        progress_text.write(
            "動画を生成中… 100%"
        )

        progress_bar.progress(
            1.0
        )

        st.success(
            "動画の生成が完了しました！"
        )

        # =========================
        # ダウンロード
        # =========================

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
            os.unlink(video_path)
        except Exception:
            pass
