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
st.write(
    "MIDIの1音ごとに動画を最初から再生し、左右反転を交互に切り替えます。"
)


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

    try:
        midi = mido.MidiFile(temp.name)
    finally:
        if os.path.exists(temp.name):
            os.remove(temp.name)

    return midi


# ============================================================
# MIDIテンポマップ
# ============================================================

def get_tempo_events(midi):

    events = []

    for track in midi.tracks:

        absolute_tick = 0

        for msg in track:

            absolute_tick += msg.time

            if msg.type == "set_tempo":

                events.append(
                    (
                        absolute_tick,
                        msg.tempo
                    )
                )

    # tick順
    events.sort(
        key=lambda x: x[0]
    )

    # 同じtickの重複を整理
    result = []

    for tick, tempo in events:

        if result and result[-1][0] == tick:

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

    return result


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

    # デフォルト120 BPM
    current_tempo = 500000

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
                current_tempo
            )

        previous_tick = tempo_tick
        current_tempo = new_tempo

    # 最後のテンポから目的tickまで
    delta_tick = (
        target_tick
        - previous_tick
    )

    if delta_tick > 0:

        seconds += mido.tick2second(
            delta_tick,
            midi.ticks_per_beat,
            current_tempo
        )

    return seconds


# ============================================================
# 選択トラックのノート取得
# ============================================================

def get_track_notes(
    midi,
    track_index
):

    track = midi.tracks[track_index]

    absolute_tick = 0

    notes = []

    for msg in track:

        absolute_tick += msg.time

        if (
            msg.type == "note_on"
            and msg.velocity > 0
        ):

            notes.append(
                {
                    "tick": absolute_tick,
                    "note": msg.note,
                    "velocity": msg.velocity
                }
            )

    return notes


# ============================================================
# ノートを秒に変換
# ============================================================

def get_notes_in_seconds(
    midi,
    track_index
):

    raw_notes = get_track_notes(
        midi,
        track_index
    )

    tempo_events = get_tempo_events(
        midi
    )

    notes = []

    for note in raw_notes:

        time = tick_to_seconds(
            midi,
            note["tick"],
            tempo_events
        )

        notes.append(
            {
                "time": time,
                "note": note["note"],
                "velocity": note["velocity"]
            }
        )

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

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

    cap.release()

    return {
        "fps": fps,
        "frames": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }


# ============================================================
# MIDI同期動画生成
#
# ★ここが今回の重要部分
#
# 各MIDIノートを境目に
# 動画を0フレームから再スタートする。
#
# 奇数番目の区間 = 通常
# 偶数番目の区間 = 左右反転
# ============================================================

def make_midi_video(
    input_path,
    output_path,
    notes
):

    # --------------------------------------------------------
    # 元動画を読み込む
    # --------------------------------------------------------

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

        cap.release()

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

    source_frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    source_duration = (
        source_frame_count / fps
    )

    cap.release()

    # --------------------------------------------------------
    # MIDIの最初の音を0秒にする
    # --------------------------------------------------------

    first_time = notes[0]["time"]

    note_times = []

    for note in notes:

        aligned_time = (
            note["time"]
            - first_time
        )

        note_times.append(
            max(0.0, aligned_time)
        )

    note_times = np.array(
        note_times,
        dtype=np.float64
    )

    # 重複・逆順防止
    note_times = np.maximum.accumulate(
        note_times
    )

    # --------------------------------------------------------
    # 最後の音以降
    #
    # 最後の区間は元動画1本分再生する
    # --------------------------------------------------------

    total_duration = (
        note_times[-1]
        + source_duration
    )

    output_frame_count = int(
        np.ceil(
            total_duration * fps
        )
    )

    # --------------------------------------------------------
    # 一時動画
    # --------------------------------------------------------

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

        raise RuntimeError(
            "動画出力を開始できませんでした。"
        )

    progress = st.progress(0)

    # ========================================================
    # 出力フレームを1枚ずつ生成
    # ========================================================

    for output_frame in range(
        output_frame_count
    ):

        # ----------------------------------------------------
        # MIDI上の現在時間
        # ----------------------------------------------------

        current_time = (
            output_frame / fps
        )

        # ----------------------------------------------------
        # 現在どの音の区間なのか
        #
        # 0 = 1音目～2音目
        # 1 = 2音目～3音目
        # 2 = 3音目～4音目
        # ...
        # ----------------------------------------------------

        segment = np.searchsorted(
            note_times,
            current_time,
            side="right"
        ) - 1

        # ----------------------------------------------------
        # まだ最初の音より前なら通常
        # ----------------------------------------------------

        if segment < 0:

            segment = 0

            segment_start = 0.0

        else:

            segment_start = (
                note_times[segment]
            )

        # ----------------------------------------------------
        # この区間の開始から何秒経ったか
        # ----------------------------------------------------

        local_time = (
            current_time
            - segment_start
        )

        # ----------------------------------------------------
        # ★ 動画を0秒から再生する
        # ----------------------------------------------------

        source_frame = int(
            local_time * fps
        )

        # ----------------------------------------------------
        # 動画の最後まで来た場合
        #
        # 最後のフレームを表示
        # ----------------------------------------------------

        if source_frame >= source_frame_count:

            source_frame = (
                source_frame_count - 1
            )

        if source_frame < 0:

            source_frame = 0

        # ----------------------------------------------------
        # 元動画から該当フレームを取得
        # ----------------------------------------------------

        cap = cv2.VideoCapture(
            input_path
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            source_frame
        )

        ret, frame = cap.read()

        cap.release()

        if not ret:

            # 読み込み失敗時はスキップ
            continue

        # ----------------------------------------------------
        # ★★★ 左右反転 ★★★
        #
        # 1音目区間 = 通常
        # 2音目区間 = 反転
        # 3音目区間 = 通常
        # 4音目区間 = 反転
        # ----------------------------------------------------

        if segment % 2 == 1:

            frame = cv2.flip(
                frame,
                1
            )

        writer.write(
            frame
        )

        if output_frame % 10 == 0:

            progress.progress(
                min(
                    output_frame
                    / max(
                        output_frame_count,
                        1
                    ),
                    1.0
                )
            )

    writer.release()

    progress.progress(1.0)

    # --------------------------------------------------------
    # 音声を元動画から追加
    # --------------------------------------------------------

    add_audio(
        temp_video,
        input_path,
        output_path
    )

    if os.path.exists(temp_video):

        os.remove(
            temp_video
        )

    return {
        "output_duration": total_duration,
        "note_count": len(notes),
        "first_note": note_times[0],
        "last_note": note_times[-1]
    }


# ============================================================
# 音声追加
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

            count = sum(
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
                f"{count} notes"
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

        if notes:

            st.write(
                "MIDIの最初の20音："
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
# ③ MIDI同期
# ============================================================

st.header("③ MIDIに同期して動画を再スタート")

st.info(
    """
MIDIの1音ごとに動画を最初から再生します。

1音目 → 通常方向で動画開始
2音目 → 0秒に戻って左右反転して再生
3音目 → 0秒に戻って通常方向で再生
4音目 → 0秒に戻って左右反転して再生
5音目 → 0秒に戻って通常方向で再生
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

    if not notes:

        st.error(
            "音が入っているMIDIトラックを選択してください。"
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
            "MIDIのテンポ・音符位置に合わせて動画を生成しています..."
        ):

            result = make_midi_video(
                video_path,
                output,
                notes
            )

        st.success(
            "🎉 完成しました！"
        )

        st.write(
            f"MIDIノート数："
            f"**{result['note_count']}**"
        )

        st.write(
            f"動画内での1音目："
            f"**0.000秒**"
        )

        st.write(
            f"最後の音："
            f"**{result['last_note']:.3f}秒**"
        )

        st.write(
            f"生成動画の長さ："
            f"**{result['output_duration']:.3f}秒**"
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
