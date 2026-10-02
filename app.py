import os
import tempfile
import subprocess
import shutil
from fractions import Fraction

import streamlit as st
import mido
import cv2
import numpy as np


# =========================================================
# ページ設定
# =========================================================

st.set_page_config(
    page_title="YTPMV向けMIDI反転ツール",
    page_icon="🎵",
    layout="wide",
)

st.title("YTPMV向けMIDI反転ツール")
st.write("スマホ勢向けの動画反転ツールです。パソコン勢の方もどうぞ。")


# =========================================================
# MIDI
# =========================================================

def build_tempo_map(mid):
    """
    MIDI全体からテンポ変更を取得する。

    戻り値:
        [(tick, tempo), ...]
    """

    tempo_events = []

    merged = mido.merge_tracks(mid.tracks)

    absolute_tick = 0

    for msg in merged:
        absolute_tick += msg.time

        if msg.type == "set_tempo":
            tempo_events.append(
                (absolute_tick, msg.tempo)
            )

    # 同じtickに複数テンポがある場合は最後のものを使用
    tempo_map = []

    for tick, tempo in tempo_events:
        if tempo_map and tempo_map[-1][0] == tick:
            tempo_map[-1] = (tick, tempo)
        else:
            tempo_map.append((tick, tempo))

    # テンポ指定がない場合
    if not tempo_map or tempo_map[0][0] != 0:
        tempo_map.insert(0, (0, 500000))

    return tempo_map


def tick_to_seconds_exact(
    tick,
    tempo_map,
    ticks_per_beat
):
    """
    MIDI tick → 秒

    floatをできるだけ使わずFractionで計算して、
    長時間再生時の累積誤差を抑える。
    """

    if tick <= 0:
        return Fraction(0, 1)

    total_seconds = Fraction(0, 1)

    current_tick = 0
    current_tempo = 500000

    for i, (tempo_tick, tempo) in enumerate(tempo_map):

        if tempo_tick > tick:
            break

        if tempo_tick > current_tick:

            delta_ticks = tempo_tick - current_tick

            total_seconds += (
                Fraction(
                    delta_ticks * current_tempo,
                    ticks_per_beat * 1_000_000
                )
            )

            current_tick = tempo_tick

        current_tempo = tempo

    # 最後のテンポ区間
    delta_ticks = tick - current_tick

    if delta_ticks > 0:
        total_seconds += Fraction(
            delta_ticks * current_tempo,
            ticks_per_beat * 1_000_000
        )

    return total_seconds


def get_track_note_ticks(track):
    """
    指定トラックのnote_on位置をtickで取得。
    """

    note_ticks = []
    absolute_tick = 0

    for msg in track:

        absolute_tick += msg.time

        if msg.type == "note_on" and msg.velocity > 0:
            note_ticks.append(absolute_tick)

    return note_ticks


def get_note_times(mid, track):
    """
    選択トラックのノート位置を秒で取得。

    最初のノートを0秒にする。
    同じタイミングの複数ノートは1つの境界として扱う。
    """

    note_ticks = get_track_note_ticks(track)

    if not note_ticks:
        return []

    tempo_map = build_tempo_map(mid)

    exact_times = []

    for tick in note_ticks:
        exact_time = tick_to_seconds_exact(
            tick,
            tempo_map,
            mid.ticks_per_beat
        )

        exact_times.append(exact_time)

    first_time = exact_times[0]

    relative_times = [
        t - first_time
        for t in exact_times
    ]

    # 同時発音を1つの境界としてまとめる
    unique_times = []

    for t in relative_times:
        if not unique_times or t != unique_times[-1]:
            unique_times.append(t)

    return unique_times


# =========================================================
# 動画FPS
# =========================================================

def get_video_fps(video_path):
    """
    ffprobeがあれば動画のFPSを正確な分数として取得。
    なければOpenCVのFPSを使用。
    """

    ffprobe = shutil.which("ffprobe")

    if ffprobe:

        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=avg_frame_rate,r_frame_rate",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            values = [
                x.strip()
                for x in result.stdout.splitlines()
                if x.strip()
            ]

            # avg_frame_rateを優先
            for value in values:

                if "/" in value:
                    num, den = value.split("/")

                    if int(den) != 0:
                        fps = Fraction(
                            int(num),
                            int(den)
                        )

                        if fps > 0:
                            return fps

                else:

                    fps_float = float(value)

                    if fps_float > 0:
                        return Fraction(
                            str(fps_float)
                        )

        except Exception:
            pass

    # ffprobeが使えない場合
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    if fps <= 0:
        fps = 30.0

    return Fraction(str(fps))


# =========================================================
# VFR対策
# =========================================================

def normalize_video_to_cfr(input_path, output_path):
    """
    VFR動画などによる長時間ズレを防ぐため、
    一度CFR動画に変換する。

    ffmpegがない場合はFalse。
    """

    ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:
        return False

    fps = get_video_fps(input_path)

    fps_text = f"{fps.numerator}/{fps.denominator}"

    command = [
        ffmpeg,
        "-y",
        "-i",
        input_path,

        # 正確なFPSでCFR化
        "-vf",
        f"fps={fps_text}",

        "-an",

        "-c:v",
        "libx264",

        "-pix_fmt",
        "yuv420p",

        output_path,
    ]

    try:

        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        return os.path.exists(output_path)

    except Exception:
        return False


# =========================================================
# 動画読み込み
# =========================================================

def read_source_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("動画を開けませんでした。")

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError("動画からフレームを読み込めませんでした。")

    fps = get_video_fps(video_path)

    height, width = frames[0].shape[:2]

    return frames, fps, width, height


# =========================================================
# MIDI時間 → 絶対フレーム位置
# =========================================================

def note_times_to_frame_positions(
    note_times,
    fps
):
    """
    MIDI時間を絶対フレーム位置に変換。

    重要:
    各区間の長さを個別にroundして足すのではなく、
    すべて0秒からの絶対位置として計算する。

    これによって長時間の累積ズレを防ぐ。
    """

    positions = []

    for exact_time in note_times:

        # 秒 × FPS
        frame_exact = (
            exact_time * fps
        )

        # Fractionのまま四捨五入
        numerator = frame_exact.numerator
        denominator = frame_exact.denominator

        frame_position = (
            numerator * 2 + denominator
        ) // (2 * denominator)

        frame_position = max(
            0,
            frame_position
        )

        if not positions:
            positions.append(frame_position)

        elif frame_position != positions[-1]:
            positions.append(frame_position)

    return positions


# =========================================================
# 共通処理
# =========================================================

def write_frame(writer, frame, mode):

    if mode == "normal":
        output_frame = frame

    elif mode == "horizontal":
        output_frame = cv2.flip(
            frame,
            1
        )

    elif mode == "vertical":
        output_frame = cv2.flip(
            frame,
            0
        )

    elif mode == "both":
        output_frame = cv2.flip(
            frame,
            -1
        )

    else:
        output_frame = frame

    writer.write(output_frame)


def create_writer(
    output_path,
    fps,
    width,
    height
):

    # OpenCVのVideoWriterはfloatを要求するので、
    # Fractionから正確な値を作る
    fps_float = (
        fps.numerator /
        fps.denominator
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps_float,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            "動画出力を開始できませんでした。"
        )

    return writer


# =========================================================
# ① 反転なし（ノートごと再生）
# =========================================================

def make_no_flip_video(
    frames,
    fps,
    note_frame_positions,
    output_path,
    update_progress
):

    source_count = len(frames)

    if not note_frame_positions:
        note_frame_positions = [0]

    # 最初のノートから開始
    note_frame_positions = [
        max(0, p)
        for p in note_frame_positions
    ]

    total_output_frames = 0

    for i in range(len(note_frame_positions)):

        if i + 1 < len(note_frame_positions):

            current_position = (
                note_frame_positions[i]
            )

            next_position = (
                note_frame_positions[i + 1]
            )

            frame_count = max(
                0,
                next_position -
                current_position
            )

        else:

            # 最後のノート以降は
            # 元動画を最後まで再生
            frame_count = source_count

        total_output_frames += frame_count

    if total_output_frames <= 0:
        raise RuntimeError(
            "出力するフレームがありません。"
        )

    writer = create_writer(
        output_path,
        fps,
        frames[0].shape[1],
        frames[0].shape[0]
    )

    written = 0

    try:

        for segment_index in range(
            len(note_frame_positions)
        ):

            if (
                segment_index + 1
                < len(note_frame_positions)
            ):

                current_position = (
                    note_frame_positions[
                        segment_index
                    ]
                )

                next_position = (
                    note_frame_positions[
                        segment_index + 1
                    ]
                )

                frame_count = max(
                    0,
                    next_position -
                    current_position
                )

            else:

                frame_count = source_count

            for frame_index in range(
                frame_count
            ):

                source_index = (
                    frame_index %
                    source_count
                )

                writer.write(
                    frames[source_index]
                )

                written += 1

                update_progress(
                    written /
                    total_output_frames
                )

    finally:
        writer.release()


# =========================================================
# ② ノートごと反転
# =========================================================

def make_note_restart_video(
    frames,
    fps,
    note_frame_positions,
    output_path,
    update_progress
):

    source_count = len(frames)

    if not note_frame_positions:
        note_frame_positions = [0]

    total_output_frames = 0

    for i in range(
        len(note_frame_positions)
    ):

        if i + 1 < len(note_frame_positions):

            frame_count = max(
                0,
                note_frame_positions[i + 1]
                - note_frame_positions[i]
            )

        else:

            frame_count = source_count

        total_output_frames += frame_count

    writer = create_writer(
        output_path,
        fps,
        frames[0].shape[1],
        frames[0].shape[0]
    )

    written = 0

    try:

        for segment_index in range(
            len(note_frame_positions)
        ):

            if (
                segment_index + 1
                < len(note_frame_positions)
            ):

                frame_count = max(
                    0,
                    note_frame_positions[
                        segment_index + 1
                    ]
                    -
                    note_frame_positions[
                        segment_index
                    ]
                )

            else:

                frame_count = source_count

            # ノート1個目は通常
            # 2個目は左右反転
            horizontal_flip = (
                segment_index % 2 == 1
            )

            for frame_index in range(
                frame_count
            ):

                source_index = (
                    frame_index %
                    source_count
                )

                frame = frames[source_index]

                if horizontal_flip:
                    frame = cv2.flip(
                        frame,
                        1
                    )

                writer.write(frame)

                written += 1

                update_progress(
                    written /
                    total_output_frames
                )

    finally:
        writer.release()


# =========================================================
# ③ 動画そのまま反転
# =========================================================

def make_normal_video(
    frames,
    fps,
    note_frame_positions,
    output_path,
    update_progress
):

    source_count = len(frames)

    # MIDI側の最後のノートまで
    # 少なくとも出力を続ける
    last_note_position = 0

    if note_frame_positions:
        last_note_position = (
            note_frame_positions[-1]
        )

    total_output_frames = max(
        source_count,
        last_note_position + 1
    )

    writer = create_writer(
        output_path,
        fps,
        frames[0].shape[1],
        frames[0].shape[0]
    )

    try:

        for output_frame in range(
            total_output_frames
        ):

            # 動画は最初から1回だけ再生
            if output_frame < source_count:
                source_frame = frames[
                    output_frame
                ]
            else:
                # 動画終了後は最後のフレームを保持
                source_frame = frames[
                    -1
                ]

            # 現在何個目のノート区間か
            note_index = np.searchsorted(
                note_frame_positions,
                output_frame,
                side="right"
            )

            # 0個目 → 通常
            # 1個目 → 左右反転
            # 2個目 → 通常
            # ...
            if note_index % 2 == 1:

                source_frame = cv2.flip(
                    source_frame,
                    1
                )

            writer.write(
                source_frame
            )

            update_progress(
                (output_frame + 1)
                /
                total_output_frames
            )

    finally:
        writer.release()


# =========================================================
# ④ 上下左右反転
# =========================================================

def make_four_direction_video(
    frames,
    fps,
    note_frame_positions,
    output_path,
    update_progress
):

    source_count = len(frames)

    if not note_frame_positions:
        note_frame_positions = [0]

    total_output_frames = 0

    for i in range(
        len(note_frame_positions)
    ):

        if i + 1 < len(note_frame_positions):

            frame_count = max(
                0,
                note_frame_positions[i + 1]
                -
                note_frame_positions[i]
            )

        else:

            frame_count = source_count

        total_output_frames += frame_count

    writer = create_writer(
        output_path,
        fps,
        frames[0].shape[1],
        frames[0].shape[0]
    )

    written = 0

    try:

        for segment_index in range(
            len(note_frame_positions)
        ):

            if (
                segment_index + 1
                < len(note_frame_positions)
            ):

                frame_count = max(
                    0,
                    note_frame_positions[
                        segment_index + 1
                    ]
                    -
                    note_frame_positions[
                        segment_index
                    ]
                )

            else:

                frame_count = source_count

            # 0: 通常
            # 1: 左右
            # 2: 上下
            # 3: 上下左右
            flip_mode = (
                segment_index % 4
            )

            for frame_index in range(
                frame_count
            ):

                source_index = (
                    frame_index %
                    source_count
                )

                frame = frames[
                    source_index
                ]

                if flip_mode == 1:

                    frame = cv2.flip(
                        frame,
                        1
                    )

                elif flip_mode == 2:

                    frame = cv2.flip(
                        frame,
                        0
                    )

                elif flip_mode == 3:

                    frame = cv2.flip(
                        frame,
                        -1
                    )

                writer.write(frame)

                written += 1

                update_progress(
                    written /
                    total_output_frames
                )

    finally:
        writer.release()


# =========================================================
# ① MIDI
# =========================================================

st.subheader("①MIDI")

midi_file = st.file_uploader(
    "MIDIファイルを選択",
    type=["mid", "midi"]
)


# =========================================================
# MIDI読み込み
# =========================================================

mid = None

if midi_file is not None:

    try:

        midi_bytes = midi_file.getvalue()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mid"
        ) as tmp:

            tmp.write(midi_bytes)
            midi_path = tmp.name

        try:

            mid = mido.MidiFile(
                midi_path
            )

        finally:

            os.remove(midi_path)

    except Exception as e:

        st.error(
            f"MIDIを読み込めませんでした: {e}"
        )
        mid = None


# =========================================================
# ② トラック
# =========================================================

selected_track = None

if mid is not None:

    st.subheader("②トラック")

    track_options = []

    for index, track in enumerate(
        mid.tracks
    ):

        note_count = len(
            get_track_note_ticks(track)
        )

        # 音数0のトラックは表示しない
        if note_count <= 0:
            continue

        track_name = ""

        for msg in track:

            if msg.type == "track_name":
                track_name = msg.name
                break

        if not track_name:
            track_name = "名前なし"

        label = (
            f"{index}: "
            f"{track_name}"
            f"（音数: {note_count}）"
        )

        track_options.append(
            (
                label,
                index
            )
        )

    if not track_options:

        st.warning(
            "ノートが入っているトラックがありません。"
        )

    else:

        selected_label = st.selectbox(
            "トラックを選択",
            [
                item[0]
                for item in track_options
            ]
        )

        for label, index in track_options:

            if label == selected_label:
                selected_track = mid.tracks[
                    index
                ]
                break


# =========================================================
# ③ 動画
# =========================================================

st.subheader("③動画")

video_file = st.file_uploader(
    "動画ファイルを選択",
    type=["mp4", "mov"]
)


# =========================================================
# ④ 反転
# =========================================================

st.subheader("④反転")

mode = st.selectbox(
    "反転方法を選択",
    [
        "反転なし（ノートごと再生）",
        "ノートごと反転",
        "動画そのまま反転",
        "上下左右反転",
    ]
)


# =========================================================
# 生成
# =========================================================

if (
    mid is not None
    and selected_track is not None
    and video_file is not None
):

    if st.button(
        "生成",
        type="primary"
    ):

        progress_text = st.empty()
        progress_bar = st.progress(0)

        def update_progress(value):

            value = min(
                1.0,
                max(
                    0.0,
                    float(value)
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

        temp_dir = tempfile.mkdtemp()

        try:

            # -------------------------------------------------
            # MIDIのノート時間
            # -------------------------------------------------

            note_times = get_note_times(
                mid,
                selected_track
            )

            if not note_times:

                st.error(
                    "選択したトラックにノートがありません。"
                )

                shutil.rmtree(
                    temp_dir,
                    ignore_errors=True
                )

                st.stop()

            # -------------------------------------------------
            # 元動画保存
            # -------------------------------------------------

            original_video_path = os.path.join(
                temp_dir,
                "original_video.mp4"
            )

            with open(
                original_video_path,
                "wb"
            ) as f:

                f.write(
                    video_file.getvalue()
                )

            # -------------------------------------------------
            # VFR対策
            # -------------------------------------------------

            normalized_video_path = os.path.join(
                temp_dir,
                "normalized.mp4"
            )

            progress_text.write(
                "動画を準備中…"
            )

            normalized = (
                normalize_video_to_cfr(
                    original_video_path,
                    normalized_video_path
                )
            )

            if normalized:

                source_video_path = (
                    normalized_video_path
                )

            else:

                source_video_path = (
                    original_video_path
                )

            # -------------------------------------------------
            # 動画読み込み
            # -------------------------------------------------

            frames, fps, width, height = (
                read_source_frames(
                    source_video_path
                )
            )

            # -------------------------------------------------
            # MIDI時間 → 絶対フレーム
            # -------------------------------------------------

            note_frame_positions = (
                note_times_to_frame_positions(
                    note_times,
                    fps
                )
            )

            if not note_frame_positions:

                st.error(
                    "MIDIノートの時間を計算できませんでした。"
                )

                shutil.rmtree(
                    temp_dir,
                    ignore_errors=True
                )

                st.stop()

            # -------------------------------------------------
            # 出力先
            # -------------------------------------------------

            output_path = os.path.join(
                temp_dir,
                "ytpmv_output.mp4"
            )

            # -------------------------------------------------
            # 生成
            # -------------------------------------------------

            if mode == "反転なし（ノートごと再生）":

                make_no_flip_video(
                    frames,
                    fps,
                    note_frame_positions,
                    output_path,
                    update_progress
                )

            elif mode == "ノートごと反転":

                make_note_restart_video(
                    frames,
                    fps,
                    note_frame_positions,
                    output_path,
                    update_progress
                )

            elif mode == "動画そのまま反転":

                make_normal_video(
                    frames,
                    fps,
                    note_frame_positions,
                    output_path,
                    update_progress
                )

            elif mode == "上下左右反転":

                make_four_direction_video(
                    frames,
                    fps,
                    note_frame_positions,
                    output_path,
                    update_progress
                )

            # -------------------------------------------------
            # 完了
            # -------------------------------------------------

            progress_text.write(
                "動画を生成中… 100%"
            )

            progress_bar.progress(1.0)

            if os.path.exists(output_path):

                st.success(
                    "動画の生成が完了しました！"
                )

                with open(
                    output_path,
                    "rb"
                ) as f:

                    video_data = f.read()

                st.download_button(
                    label="生成した動画を保存",
                    data=video_data,
                    file_name="ytpmv_output.mp4",
                    mime="video/mp4",
                )

            else:

                st.error(
                    "動画の生成に失敗しました。"
                )

        except Exception as e:

            st.error(
                f"動画生成中にエラーが発生しました: {e}"
            )

        finally:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )
