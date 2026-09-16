import os
import tempfile
import streamlit as st
import mido
import cv2
import numpy as np


# =========================================================
# MIDI読み込み
# =========================================================

def load_midi(file_path):
    return mido.MidiFile(file_path)


# =========================================================
# トラック名を取得
# =========================================================

def get_track_name(track, index):
    for msg in track:
        if msg.type == "track_name":
            name = msg.name

            # bytesになっている場合に対応
            if isinstance(name, bytes):
                try:
                    name = name.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        name = name.decode("cp932")
                    except UnicodeDecodeError:
                        name = ""

            # 空の名前ならデフォルト名
            if name and str(name).strip():
                return str(name).strip()

    return f"Track {index}"


# =========================================================
# トラックの音数を取得
# =========================================================

def count_notes(track):
    count = 0

    for msg in track:
        if (
            msg.type == "note_on"
            and msg.velocity > 0
        ):
            count += 1

    return count


# =========================================================
# MIDIテンポ情報
# =========================================================

def get_tempo_events(mid):
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

    tempo_events.sort(
        key=lambda x: x[0]
    )

    result = []

    for tick, tempo in tempo_events:

        if (
            result
            and result[-1][0] == tick
        ):
            result[-1] = (
                tick,
                tempo
            )
        else:
            result.append(
                (
                    tick,
                    tempo
                )
            )

    # テンポ情報がない場合
    if not result:
        result = [
            (0, 500000)
        ]

    # 0 tickから始まっていない場合
    if result[0][0] != 0:
        result.insert(
            0,
            (0, 500000)
        )

    return result


# =========================================================
# MIDI tick → 秒
# =========================================================

def tick_to_seconds(
    target_tick,
    ticks_per_beat,
    tempo_events
):

    seconds = 0.0

    previous_tick = 0
    current_tempo = 500000

    for tick, tempo in tempo_events:

        if tick > target_tick:
            break

        delta_tick = (
            tick - previous_tick
        )

        seconds += (
            delta_tick
            * current_tempo
            / 1_000_000
            / ticks_per_beat
        )

        previous_tick = tick
        current_tempo = tempo

    delta_tick = (
        target_tick - previous_tick
    )

    seconds += (
        delta_tick
        * current_tempo
        / 1_000_000
        / ticks_per_beat
    )

    return seconds


# =========================================================
# MIDIノート取得
# =========================================================

def get_notes_in_seconds(
    mid,
    track_index
):

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

            notes.append(
                {
                    "note": msg.note,
                    "time": time_sec
                }
            )

    return notes


# =========================================================
# ノート時間
# 最初のノートを0秒にする
# =========================================================

def get_note_times(notes):

    first_midi_time = (
        notes[0]["time"]
    )

    note_times = [
        max(
            0.0,
            note["time"]
            - first_midi_time
        )
        for note in notes
    ]

    return np.maximum.accumulate(
        note_times
    )


# =========================================================
# 動画情報取得
# =========================================================

def get_video_info(cap):

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

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        frame_count / fps
    )

    return (
        fps,
        width,
        height,
        frame_count,
        duration
    )


# =========================================================
# 反転なし
#
# ノートごとに動画を0秒から再生
# 反転は一切しない
# =========================================================

def make_no_flip_video(
    input_path,
    notes,
    output_path,
    progress_callback=None
):

    cap = cv2.VideoCapture(
        input_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            "動画を開けませんでした"
        )

    (
        fps,
        width,
        height,
        source_frame_count,
        source_duration
    ) = get_video_info(cap)

    note_times = get_note_times(
        notes
    )

    total_duration = (
        note_times[-1]
        + source_duration
    )

    output_frame_count = int(
        np.ceil(
            total_duration * fps
        )
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

        segment_start = (
            note_times[segment]
        )

        local_time = (
            current_time
            - segment_start
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

        # 反転なし
        out.write(frame)

        if progress_callback:
            progress_callback(
                (output_frame + 1)
                / output_frame_count
            )

    cap.release()
    out.release()


# =========================================================
# ノートごと反転
#
# ノート1 → 通常
# ノート2 → 左右反転
# ノート3 → 通常
# ノート4 → 左右反転
# ...
#
# ノートごとに動画を0秒から再生
# =========================================================

def make_note_restart_video(
    input_path,
    notes,
    output_path,
    progress_callback=None
):

    cap = cv2.VideoCapture(
        input_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            "動画を開けませんでした"
        )

    (
        fps,
        width,
        height,
        source_frame_count,
        source_duration
    ) = get_video_info(cap)

    note_times = get_note_times(
        notes
    )

    total_duration = (
        note_times[-1]
        + source_duration
    )

    output_frame_count = int(
        np.ceil(
            total_duration * fps
        )
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

        segment_start = (
            note_times[segment]
        )

        local_time = (
            current_time
            - segment_start
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

        # 奇数区間だけ左右反転
        if segment % 2 == 1:
            frame = cv2.flip(
                frame,
                1
            )

        out.write(frame)

        if progress_callback:
            progress_callback(
                (output_frame + 1)
                / output_frame_count
            )

    cap.release()
    out.release()


# =========================================================
# 動画そのまま反転
#
# 動画は最初から最後まで1回だけ再生
#
# ノート1 → 通常
# ノート2 → 左右反転
# ノート3 → 通常
# ...
#
# 動画終了後は最後のフレームを保持
# =========================================================

def make_normal_video(
    input_path,
    notes,
    output_path,
    progress_callback=None
):

    cap = cv2.VideoCapture(
        input_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            "動画を開けませんでした"
        )

    (
        fps,
        width,
        height,
        source_frame_count,
        source_duration
    ) = get_video_info(cap)

    note_times = get_note_times(
        notes
    )

    total_duration = max(
        source_duration,
        note_times[-1]
    )

    output_frame_count = int(
        np.ceil(
            total_duration * fps
        )
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

        while (
            note_index + 1
            < len(note_times)
            and current_time
            >= note_times[
                note_index + 1
            ]
        ):

            note_index += 1

        # 元動画を1回だけ再生
        if (
            output_frame
            < source_frame_count
        ):

            ret, frame = cap.read()

            if ret:
                last_frame = frame

        # 動画終了後は最後のフレーム
        if last_frame is None:
            continue

        frame = last_frame.copy()

        # 奇数ノートで左右反転
        if note_index % 2 == 1:
            frame = cv2.flip(
                frame,
                1
            )

        out.write(frame)

        if progress_callback:
            progress_callback(
                (output_frame + 1)
                / output_frame_count
            )

    cap.release()
    out.release()


# =========================================================
# 上下左右反転
#
# ノート1 → 通常
# ノート2 → 左右
# ノート3 → 上下
# ノート4 → 上下左右
# ノート5 → 通常
# ...
#
# ノートごとに動画を0秒から再生
# =========================================================

def make_four_direction_video(
    input_path,
    notes,
    output_path,
    progress_callback=None
):

    cap = cv2.VideoCapture(
        input_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            "動画を開けませんでした"
        )

    (
        fps,
        width,
        height,
        source_frame_count,
        source_duration
    ) = get_video_info(cap)

    note_times = get_note_times(
        notes
    )

    total_duration = (
        note_times[-1]
        + source_duration
    )

    output_frame_count = int(
        np.ceil(
            total_duration * fps
        )
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

        segment_start = (
            note_times[segment]
        )

        local_time = (
            current_time
            - segment_start
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

        mode = segment % 4

        if mode == 0:

            # 通常
            pass

        elif mode == 1:

            # 左右反転
            frame = cv2.flip(
                frame,
                1
            )

        elif mode == 2:

            # 上下反転
            frame = cv2.flip(
                frame,
                0
            )

        elif mode == 3:

            # 上下左右反転
            frame = cv2.flip(
                frame,
                -1
            )

        out.write(frame)

        if progress_callback:
            progress_callback(
                (output_frame + 1)
                / output_frame_count
            )

    cap.release()
    out.release()


# =========================================================
# Streamlit設定
# =========================================================

st.set_page_config(
    page_title="YTPMV向けMIDI反転ツール"
)

st.title(
    "YTPMV向けMIDI反転ツール"
)


# =========================================================
# ① MIDI
# =========================================================

midi_file = st.file_uploader(
    "① MIDIファイルを選択",
    type=[
        "mid",
        "midi"
    ]
)

notes = None
track_index = None


if midi_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mid"
    ) as tmp:

        tmp.write(
            midi_file.read()
        )

        midi_path = tmp.name

    try:

        mid = load_midi(
            midi_path
        )

        # =================================================
        # 音が1つ以上あるトラックだけを作る
        # =================================================

        track_options = []

        for i, track in enumerate(
            mid.tracks
        ):

            note_count = count_notes(
                track
            )

            # 音数0なら完全に除外
            if note_count <= 0:
                continue

            track_name = get_track_name(
                track,
                i
            )

            display_name = (
                f"{i}: {track_name}"
                f"（音数: {note_count}）"
            )

            track_options.append(
                (
                    i,
                    display_name,
                    note_count
                )
            )

        # =================================================
        # 音のあるトラックがある場合
        # =================================================

        if track_options:

            selected_option = st.selectbox(
                "MIDIトラックを選択",
                track_options,
                format_func=lambda x: x[1]
            )

            track_index = (
                selected_option[0]
            )

            notes = get_notes_in_seconds(
                mid,
                track_index
            )

        else:

            st.warning(
                "音符が入っているMIDIトラックがありません。"
            )

            notes = None

    except Exception as e:

        st.error(
            f"MIDIの読み込みに失敗しました：{e}"
        )

    finally:

        if os.path.exists(
            midi_path
        ):
            os.remove(
                midi_path
            )


# =========================================================
# ② 動画
# =========================================================

video_file = st.file_uploader(
    "② 動画ファイルを選択",
    type=[
        "mp4",
        "mov"
    ]
)


# =========================================================
# ③ 生成
# =========================================================

st.subheader(
    "③ 生成"
)


mode = st.radio(
    "反転方式",
    [
        "反転なし（ノートごと再生）",
        "ノートごと反転",
        "動画そのまま反転",
        "上下左右反転"
    ]
)


# =========================================================
# 生成ボタン
# =========================================================

if (
    midi_file
    and video_file
    and notes
):

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

        progress_bar = st.progress(
            0
        )

        percent_text = st.empty()


        def update_progress(
            value
        ):

            percent = int(
                value * 100
            )

            progress_bar.progress(
                min(
                    percent,
                    100
                )
            )

            percent_text.write(
                f"生成中… {percent}%"
            )


        try:

            # =============================================
            # 反転なし
            # =============================================

            if (
                mode
                == "反転なし（ノートごと再生）"
            ):

                make_no_flip_video(
                    input_video_path,
                    notes,
                    output_path,
                    update_progress
                )


            # =============================================
            # ノートごと左右反転
            # =============================================

            elif (
                mode
                == "ノートごと反転"
            ):

                make_note_restart_video(
                    input_video_path,
                    notes,
                    output_path,
                    update_progress
                )


            # =============================================
            # 動画そのまま反転
            # =============================================

            elif (
                mode
                == "動画そのまま反転"
            ):

                make_normal_video(
                    input_video_path,
                    notes,
                    output_path,
                    update_progress
                )


            # =============================================
            # 上下左右反転
            # =============================================

            elif (
                mode
                == "上下左右反転"
            ):

                make_four_direction_video(
                    input_video_path,
                    notes,
                    output_path,
                    update_progress
                )


            # =============================================
            # 完了
            # =============================================

            progress_bar.progress(
                100
            )

            percent_text.write(
                "生成完了！ 100%"
            )

            st.video(
                output_path
            )


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
