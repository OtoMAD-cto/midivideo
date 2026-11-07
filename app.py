import os
import tempfile
from fractions import Fraction
import shutil

import streamlit as st
import mido
import cv2


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

def get_track_note_ticks(track):
    """
    トラック内のnote_on位置を
    トラック先頭からの絶対tickで取得。

    velocity=0のnote_onはnote_off扱いなので除外。
    """

    note_ticks = []
    absolute_tick = 0

    for msg in track:

        absolute_tick += msg.time

        if (
            msg.type == "note_on"
            and msg.velocity > 0
        ):
            note_ticks.append(
                absolute_tick
            )

    return note_ticks


def get_note_times_by_bpm(
    mid,
    track,
    bpm
):
    """
    MIDI内部のテンポ情報は使用しない。

    ユーザーが入力したBPMだけを使用する。

    重要：
    トラック内の最初のノートを0秒にはしない。

    MIDI全体の先頭からのtick位置を維持することで、
    複数トラックを別々に生成して同時再生した場合でも
    MIDI上のタイミングを一致させる。
    """

    note_ticks = get_track_note_ticks(
        track
    )

    if not note_ticks:
        return []

    ticks_per_beat = mid.ticks_per_beat

    bpm_fraction = Fraction(
        str(bpm)
    )

    # 1拍あたりの秒数
    seconds_per_beat = (
        Fraction(60, 1)
        /
        bpm_fraction
    )

    note_times = []

    for tick in note_ticks:

        # ★重要
        # 最初のノートを引かない
        # MIDI全体の先頭からの位置をそのまま使う
        beat_position = Fraction(
            tick,
            ticks_per_beat
        )

        exact_seconds = (
            beat_position
            *
            seconds_per_beat
        )

        note_times.append(
            exact_seconds
        )

    # 同じタイミングのノートは1つの境界にする
    unique_times = []

    for time in note_times:

        if (
            not unique_times
            or time != unique_times[-1]
        ):
            unique_times.append(
                time
            )

    return unique_times


# =========================================================
# 動画読み込み
# =========================================================

def read_source_frames(
    video_path
):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            "動画を開けませんでした。"
        )

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frames.append(
            frame
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    cap.release()

    if not frames:
        raise RuntimeError(
            "動画からフレームを読み込めませんでした。"
        )

    if fps <= 0:
        fps = 30.0

    return (
        frames,
        Fraction(str(fps))
    )


# =========================================================
# MIDI時間 → 絶対フレーム位置
# =========================================================

def note_times_to_frame_positions(
    note_times,
    fps
):
    """
    MIDI全体の先頭からの絶対時間を
    絶対フレーム位置へ変換。

    各区間を個別に丸めない。
    """

    positions = []

    for exact_time in note_times:

        frame_exact = (
            exact_time
            *
            fps
        )

        numerator = (
            frame_exact.numerator
        )

        denominator = (
            frame_exact.denominator
        )

        # 四捨五入
        frame_position = (
            numerator * 2
            +
            denominator
        ) // (
            2 * denominator
        )

        frame_position = max(
            0,
            frame_position
        )

        if (
            not positions
            or frame_position != positions[-1]
        ):
            positions.append(
                frame_position
            )

    return positions


# =========================================================
# VideoWriter
# =========================================================

def create_writer(
    output_path,
    fps,
    width,
    height
):

    fps_float = (
        fps.numerator
        /
        fps.denominator
    )

    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps_float,
        (
            width,
            height
        )
    )

    if not writer.isOpened():
        raise RuntimeError(
            "動画出力を開始できませんでした。"
        )

    return writer


# =========================================================
# ノート区間の総フレーム数
# =========================================================

def get_total_restart_frames(
    note_frame_positions,
    source_count
):
    """
    ノートごと再生方式の総出力フレーム数。

    最初のノートが0秒ではない場合も、
    MIDI全体のタイミングを維持するため、
    最初のノートまでの時間を確保する。
    """

    if not note_frame_positions:
        return source_count

    last_note_position = (
        note_frame_positions[-1]
    )

    return (
        last_note_position
        +
        source_count
    )


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

    total_output_frames = (
        get_total_restart_frames(
            note_frame_positions,
            source_count
        )
    )

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

        # =================================================
        # MIDI全体の先頭から
        # 最初のノートまで
        #
        # 最初のノートが0秒でない場合でも
        # タイミングを維持する。
        # =================================================

        for output_frame in range(
            total_output_frames
        ):

            # ---------------------------------------------
            # まだ最初のノートに到達していない
            # ---------------------------------------------

            if (
                output_frame
                <
                note_frame_positions[0]
            ):

                source_index = min(
                    output_frame,
                    source_count - 1
                )

            else:

                # -----------------------------------------
                # 現在のノート区間を検索
                # -----------------------------------------

                note_index = 0

                for position in note_frame_positions:

                    if output_frame >= position:

                        note_index += 1

                    else:

                        break

                current_note_index = (
                    note_index - 1
                )

                segment_start = (
                    note_frame_positions[
                        current_note_index
                    ]
                )

                frame_in_segment = (
                    output_frame
                    -
                    segment_start
                )

                # 動画が終わったら最後のフレームで静止
                source_index = min(
                    frame_in_segment,
                    source_count - 1
                )

            frame = frames[
                source_index
            ]

            writer.write(
                frame
            )

            written += 1

            update_progress(
                written
                /
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

    total_output_frames = (
        get_total_restart_frames(
            note_frame_positions,
            source_count
        )
    )

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

        for output_frame in range(
            total_output_frames
        ):

            # =================================================
            # 最初のノートより前
            # =================================================

            if (
                output_frame
                <
                note_frame_positions[0]
            ):

                source_index = min(
                    output_frame,
                    source_count - 1
                )

                frame = frames[
                    source_index
                ]

            else:

                # =================================================
                # 現在のノート区間
                # =================================================

                note_index = 0

                for position in note_frame_positions:

                    if output_frame >= position:

                        note_index += 1

                    else:

                        break

                current_note_index = (
                    note_index - 1
                )

                segment_start = (
                    note_frame_positions[
                        current_note_index
                    ]
                )

                frame_in_segment = (
                    output_frame
                    -
                    segment_start
                )

                # 動画が終わったら最後のフレームで静止
                source_index = min(
                    frame_in_segment,
                    source_count - 1
                )

                frame = frames[
                    source_index
                ]

                # =================================================
                # ノートごとに左右反転
                # =================================================

                horizontal_flip = (
                    current_note_index % 2 == 1
                )

                if horizontal_flip:

                    frame = cv2.flip(
                        frame,
                        1
                    )

            writer.write(
                frame
            )

            written += 1

            update_progress(
                written
                /
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

    if not note_frame_positions:

        note_frame_positions = [0]

    last_note_position = (
        note_frame_positions[-1]
    )

    # MIDI最後のノートまで
    # さらに元動画1本分を確保
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

            # =================================================
            # 動画は最初から最後まで1回だけ再生
            # 終了後は最後のフレームで静止
            # =================================================

            source_index = min(
                output_frame,
                source_count - 1
            )

            frame = frames[
                source_index
            ]

            # =================================================
            # 現在のノート区間
            # =================================================

            note_index = 0

            for position in note_frame_positions:

                if output_frame >= position:

                    note_index += 1

                else:

                    break

            # =================================================
            # ノート境界ごとに左右反転
            # =================================================

            if note_index % 2 == 1:

                frame = cv2.flip(
                    frame,
                    1
                )

            writer.write(
                frame
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

    total_output_frames = (
        get_total_restart_frames(
            note_frame_positions,
            source_count
        )
    )

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

        for output_frame in range(
            total_output_frames
        ):

            # =================================================
            # 最初のノートより前
            # =================================================

            if (
                output_frame
                <
                note_frame_positions[0]
            ):

                source_index = min(
                    output_frame,
                    source_count - 1
                )

                frame = frames[
                    source_index
                ]

            else:

                # =================================================
                # 現在のノート区間
                # =================================================

                note_index = 0

                for position in note_frame_positions:

                    if output_frame >= position:

                        note_index += 1

                    else:

                        break

                current_note_index = (
                    note_index - 1
                )

                segment_start = (
                    note_frame_positions[
                        current_note_index
                    ]
                )

                frame_in_segment = (
                    output_frame
                    -
                    segment_start
                )

                # 動画終了後は最後のフレームで静止
                source_index = min(
                    frame_in_segment,
                    source_count - 1
                )

                frame = frames[
                    source_index
                ]

                # =================================================
                # 4方向
                #
                # 0 = 通常
                # 1 = 左右
                # 2 = 上下
                # 3 = 上下左右
                # =================================================

                flip_mode = (
                    current_note_index % 4
                )

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

            writer.write(
                frame
            )

            written += 1

            update_progress(
                written
                /
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
    type=[
        "mid",
        "midi"
    ]
)


# =========================================================
# MIDI読み込み
# =========================================================

mid = None

if midi_file is not None:

    try:

        midi_bytes = (
            midi_file.getvalue()
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mid"
        ) as tmp:

            tmp.write(
                midi_bytes
            )

            midi_path = tmp.name

        try:

            mid = mido.MidiFile(
                midi_path
            )

        finally:

            os.remove(
                midi_path
            )

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
            get_track_note_ticks(
                track
            )
        )

        # ノート0個のトラックは表示しない
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

                selected_track = (
                    mid.tracks[index]
                )

                break


# =========================================================
# ③ 動画
# =========================================================

st.subheader("③動画")

video_file = st.file_uploader(
    "動画ファイルを選択",
    type=[
        "mp4",
        "mov"
    ]
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
# ⑤ BPM
# =========================================================

st.subheader("⑤BPM")

bpm = st.number_input(
    "BPMを入力",
    min_value=1.0,
    max_value=1000.0,
    value=120.0,
    step=1.0
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

        progress_bar = st.progress(
            0
        )

        def update_progress(
            value
        ):

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

            # =================================================
            # BPMからMIDIノート時間を計算
            # =================================================

            note_times = (
                get_note_times_by_bpm(
                    mid,
                    selected_track,
                    bpm
                )
            )

            if not note_times:

                st.error(
                    "選択したトラックにノートがありません。"
                )

                st.stop()

            # =================================================
            # 動画保存
            # =================================================

            video_path = os.path.join(
                temp_dir,
                "input_video.mp4"
            )

            with open(
                video_path,
                "wb"
            ) as f:

                f.write(
                    video_file.getvalue()
                )

            # =================================================
            # 動画読み込み
            # =================================================

            progress_text.write(
                "動画を読み込んでいます…"
            )

            frames, fps = (
                read_source_frames(
                    video_path
                )
            )

            # =================================================
            # MIDI時間 → 絶対フレーム
            # =================================================

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

                st.stop()

            # =================================================
            # 出力
            # =================================================

            output_path = os.path.join(
                temp_dir,
                "ytpmv_output.mp4"
            )

            # =================================================
            # 生成
            # =================================================

            if (
                mode
                ==
                "反転なし（ノートごと再生）"
            ):

                make_no_flip_video(
                    frames,
                    fps,
                    note_frame_positions,
                    output_path,
                    update_progress
                )

            elif (
                mode
                ==
                "ノートごと反転"
            ):

                make_note_restart_video(
                    frames,
                    fps,
                    note_frame_positions,
                    output_path,
                    update_progress
                )

            elif (
                mode
                ==
                "動画そのまま反転"
            ):

                make_normal_video(
                    frames,
                    fps,
                    note_frame_positions,
                    output_path,
                    update_progress
                )

            elif (
                mode
                ==
                "上下左右反転"
            ):

                make_four_direction_video(
                    frames,
                    fps,
                    note_frame_positions,
                    output_path,
                    update_progress
                )

            # =================================================
            # 完了
            # =================================================

            progress_text.write(
                "動画を生成中… 100%"
            )

            progress_bar.progress(
                1.0
            )

            if os.path.exists(
                output_path
            ):

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
