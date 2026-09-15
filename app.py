import os
import math
import tempfile
import subprocess

import cv2
import mido
import numpy as np
import streamlit as st


# =========================================================
# ページ設定
# =========================================================

st.set_page_config(
    page_title="YTPMV Sync Tool",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 YTPMV Sync Tool")
st.caption("MIDIのノートタイミングに映像を1音単位で同期")


# =========================================================
# MIDI処理
# =========================================================

def get_midi_tracks(mid):
    """
    MIDIの各トラックについて、
    ノートONの絶対時間(秒)を計算する。
    
    mido.merge_tracks() を使わず、
    選択されたトラック単独のテンポ情報を考慮する。
    """

    tracks = []

    # MIDIのテンポは通常 Track 0 に存在することが多い。
    # ただし、SMFによっては別トラックの場合もあるため、
    # 全トラックから tempo change を集める。
    tempo_events = []

    for track_index, track in enumerate(mid.tracks):
        absolute_tick = 0

        for msg in track:
            absolute_tick += msg.time

            if msg.type == "set_tempo":
                tempo_events.append(
                    (absolute_tick, msg.tempo)
                )

    # tick順に整理
    tempo_events.sort(key=lambda x: x[0])

    # 初期テンポ
    default_tempo = 500000  # 120 BPM

    def tick_to_seconds(target_tick):
        """
        MIDI tick → 実時間(秒)
        テンポ変更を考慮する。
        """

        if target_tick <= 0:
            return 0.0

        current_tick = 0
        current_tempo = default_tempo
        seconds = 0.0

        for tempo_tick, tempo in tempo_events:

            if tempo_tick > target_tick:
                break

            if tempo_tick > current_tick:
                delta_ticks = tempo_tick - current_tick

                seconds += mido.tick2second(
                    delta_ticks,
                    mid.ticks_per_beat,
                    current_tempo
                )

                current_tick = tempo_tick

            current_tempo = tempo

        if target_tick > current_tick:
            delta_ticks = target_tick - current_tick

            seconds += mido.tick2second(
                delta_ticks,
                mid.ticks_per_beat,
                current_tempo
            )

        return seconds

    for index, track in enumerate(mid.tracks):

        absolute_tick = 0
        notes = []

        for msg in track:

            absolute_tick += msg.time

            if msg.type == "note_on" and msg.velocity > 0:

                seconds = tick_to_seconds(
                    absolute_tick
                )

                notes.append({
                    "time": seconds,
                    "note": msg.note,
                    "velocity": msg.velocity
                })

        tracks.append({
            "index": index,
            "name": track.name if hasattr(track, "name") else f"Track {index}",
            "notes": notes
        })

    return tracks


# =========================================================
# 動画情報
# =========================================================

def get_video_info(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("動画を開けませんでした。")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "frames": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }


# =========================================================
# 動画編集
# =========================================================

def process_video(
    input_path,
    output_path,
    notes,
    mode,
    flash_duration,
    invert_strength
):
    """
    MIDIノートタイミングに合わせて映像を変化させる。

    mode:
        toggle
            ノートが鳴るたびに反転ON/OFF

        flash
            ノート発生直後だけ反転

    invert_strength:
        0.0 = 元映像
        1.0 = 完全反転
    """

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("動画を開けませんでした。")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # H264
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    temp_video = output_path + ".temp.mp4"

    writer = cv2.VideoWriter(
        temp_video,
        fourcc,
        fps,
        (width, height)
    )

    # ノート時刻をnumpy化
    note_times = np.array(
        [n["time"] for n in notes],
        dtype=np.float64
    )

    total_frames = frame_count

    progress = st.progress(0)

    toggle_state = False
    note_index = 0

    for frame_number in range(frame_count):

        ret, frame = cap.read()

        if not ret:
            break

        # -------------------------------------------------
        # フレームの正確な時間
        # -------------------------------------------------

        current_time = frame_number / fps

        # -------------------------------------------------
        # 現在時刻以前に発生したノートを処理
        # -------------------------------------------------

        while (
            note_index < len(note_times)
            and note_times[note_index] <= current_time
        ):

            if mode == "toggle":
                toggle_state = not toggle_state

            note_index += 1

        # -------------------------------------------------
        # エフェクト判定
        # -------------------------------------------------

        effect = False

        if mode == "toggle":

            effect = toggle_state

        elif mode == "flash":

            # 現在時刻の直前に鳴ったノートを探す
            left = np.searchsorted(
                note_times,
                current_time - flash_duration,
                side="left"
            )

            right = np.searchsorted(
                note_times,
                current_time,
                side="right"
            )

            if right > left:
                effect = True

        # -------------------------------------------------
        # 反転
        # -------------------------------------------------

        if effect:

            inverted = cv2.bitwise_not(frame)

            if invert_strength >= 1.0:

                frame = inverted

            else:

                frame = cv2.addWeighted(
                    frame,
                    1.0 - invert_strength,
                    inverted,
                    invert_strength,
                    0
                )

        writer.write(frame)

        # -------------------------------------------------
        # Streamlit進捗
        # -------------------------------------------------

        if frame_number % 10 == 0:

            progress.progress(
                min(
                    frame_number / total_frames,
                    1.0
                )
            )

    cap.release()
    writer.release()

    progress.progress(1.0)

    # -----------------------------------------------------
    # OpenCVで作った動画には音声がないため、
    # FFmpegで元動画の音声を戻す
    # -----------------------------------------------------

    final_temp = output_path + ".final.mp4"

    command = [
        "ffmpeg",
        "-y",

        "-i",
        temp_video,

        "-i",
        input_path,

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

        "-c:a",
        "aac",

        "-b:a",
        "192k",

        "-shortest",

        final_temp
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            "FFmpegで音声を結合できませんでした。\n\n"
            + result.stderr[-3000:]
        )

    os.replace(
        final_temp,
        output_path
    )

    if os.path.exists(temp_video):
        os.remove(temp_video)


# =========================================================
# UI
# =========================================================

st.header("1. MIDIファイル")

midi_file = st.file_uploader(
    "MIDI (.mid / .midi)",
    type=["mid", "midi"]
)


if midi_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mid"
    ) as f:

        f.write(midi_file.read())
        midi_path = f.name

    try:

        mid = mido.MidiFile(midi_path)

        tracks = get_midi_tracks(mid)

        st.success(
            f"MIDI読み込み成功：{len(tracks)}トラック"
        )

        track_options = []

        for track in tracks:

            label = (
                f"Track {track['index']}"
                f" — {track['name']}"
                f" — {len(track['notes'])} notes"
            )

            track_options.append(label)

        selected_index = st.selectbox(
            "同期に使用するトラック",
            range(len(track_options)),
            format_func=lambda x: track_options[x]
        )

        selected_track = tracks[selected_index]

        notes = selected_track["notes"]

        if notes:

            st.write(
                f"🎵 ノート数：**{len(notes)}**"
            )

            st.write(
                f"最初のノート："
                f"**{notes[0]['time']:.6f} 秒**"
            )

            st.write(
                f"最後のノート："
                f"**{notes[-1]['time']:.6f} 秒**"
            )

        else:

            st.warning(
                "このトラックにはNote Onイベントがありません。"
            )

    except Exception as e:

        st.error(
            f"MIDI読み込みエラー：{e}"
        )

        notes = []


st.header("2. 動画ファイル")

video_file = st.file_uploader(
    "動画 (.mp4 / .mov)",
    type=["mp4", "mov"]
)


video_info = None
video_path = None

if video_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(video_file.name)[1]
    ) as f:

        f.write(video_file.read())
        video_path = f.name

    try:

        video_info = get_video_info(
            video_path
        )

        st.success("動画読み込み成功")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "FPS",
            f"{video_info['fps']:.3f}"
        )

        col2.metric(
            "解像度",
            f"{video_info['width']} × {video_info['height']}"
        )

        col3.metric(
            "長さ",
            f"{video_info['duration']:.2f} 秒"
        )

    except Exception as e:

        st.error(
            f"動画読み込みエラー：{e}"
        )


st.header("3. YTPMV同期設定")

mode = st.radio(
    "映像変化モード",
    [
        "toggle",
        "flash"
    ],
    format_func=lambda x: {
        "toggle": "🔄 トグル — 音が鳴るたびに反転ON/OFF",
        "flash": "⚡ フラッシュ — 音が鳴った瞬間だけ反転"
    }[x]
)

invert_strength = st.slider(
    "反転強度",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05
)

flash_duration = 0.05

if mode == "flash":

    flash_duration = st.slider(
        "フラッシュ時間（秒）",
        min_value=0.005,
        max_value=0.5,
        value=0.05,
        step=0.005
    )

st.divider()


# =========================================================
# 出力
# =========================================================

if st.button(
    "🎬 YTPMV動画を生成",
    type="primary",
    use_container_width=True
):

    if not midi_file:

        st.error("MIDIファイルをアップロードしてください。")

    elif not video_file:

        st.error("動画ファイルをアップロードしてください。")

    elif not notes:

        st.error("使用可能なMIDIノートがありません。")

    else:

        output_path = os.path.join(
            tempfile.gettempdir(),
            "ytpmv_output.mp4"
        )

        try:

            with st.spinner(
                "MIDIのノートタイミングに合わせて映像を処理しています..."
            ):

                process_video(
                    video_path,
                    output_path,
                    notes,
                    mode,
                    flash_duration,
                    invert_strength
                )

            st.success(
                "🎉 動画生成完了！"
            )

            st.video(
                output_path
            )

            with open(
                output_path,
                "rb"
            ) as f:

                st.download_button(
                    "⬇️ MP4を保存",
                    data=f,
                    file_name="ytpmv_output.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"動画生成中にエラーが発生しました。\n\n{e}"
            )
