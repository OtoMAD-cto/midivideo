import os
import tempfile
import subprocess

import cv2
import mido
import numpy as np
import streamlit as st


# ============================================================
# ページ設定
# ============================================================

st.set_page_config(
    page_title="YTPMV Sync Tool",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 YTPMV Sync Tool")
st.write("MIDIのノートに合わせて動画を反転させるツール")


# ============================================================
# MIDIからノートの発音時刻を取得
# ============================================================

def get_midi_notes(mid):
    """
    MIDIからNote Onのタイミングを秒単位で取得する。
    テンポ変更にも対応。
    """

    notes = []

    # MIDI全体を時間順に統合
    merged = mido.merge_tracks(mid.tracks)

    current_time = 0.0

    # 最初のテンポ
    tempo = 500000  # 120 BPM

    for msg in merged:

        # delta timeを秒に変換
        current_time += mido.tick2second(
            msg.time,
            mid.ticks_per_beat,
            tempo
        )

        # テンポ変更
        if msg.type == "set_tempo":
            tempo = msg.tempo

        # Note On
        if msg.type == "note_on" and msg.velocity > 0:

            notes.append({
                "time": current_time,
                "note": msg.note,
                "velocity": msg.velocity
            })

    return notes


# ============================================================
# MIDIトラック一覧
# ============================================================

def get_tracks(mid):

    tracks = []

    for index, track in enumerate(mid.tracks):

        # トラック名
        name = None

        for msg in track:

            if msg.type == "track_name":

                name = msg.name
                break

        if not name:
            name = f"Track {index}"

        # このトラックのノートを取得
        notes = []

        absolute_tick = 0

        for msg in track:

            absolute_tick += msg.time

            if msg.type == "note_on" and msg.velocity > 0:

                notes.append({
                    "tick": absolute_tick,
                    "note": msg.note,
                    "velocity": msg.velocity
                })

        tracks.append({
            "index": index,
            "name": name,
            "notes": notes
        })

    return tracks


# ============================================================
# 選択したトラックのノートを秒に変換
# ============================================================

def convert_track_notes_to_seconds(mid, track_index):

    track = mid.tracks[track_index]

    # 全MIDIからテンポ情報を取得
    tempo_events = []

    for t in mid.tracks:

        absolute_tick = 0

        for msg in t:

            absolute_tick += msg.time

            if msg.type == "set_tempo":

                tempo_events.append(
                    (absolute_tick, msg.tempo)
                )

    tempo_events.sort(
        key=lambda x: x[0]
    )

    # デフォルト120 BPM
    current_tempo = 500000

    notes = []

    absolute_tick = 0
    previous_tick = 0
    elapsed_seconds = 0.0

    tempo_index = 0

    for msg in track:

        absolute_tick += msg.time

        # 現在位置までにあるテンポ変更を処理
        while (
            tempo_index < len(tempo_events)
            and tempo_events[tempo_index][0] <= absolute_tick
        ):

            tempo_tick, new_tempo = tempo_events[tempo_index]

            # 前回tickからテンポ変更位置まで
            delta_ticks = (
                tempo_tick - previous_tick
            )

            if delta_ticks > 0:

                elapsed_seconds += mido.tick2second(
                    delta_ticks,
                    mid.ticks_per_beat,
                    current_tempo
                )

            previous_tick = tempo_tick
            current_tempo = new_tempo

            tempo_index += 1

        # 現在位置までの時間
        delta_ticks = (
            absolute_tick - previous_tick
        )

        if delta_ticks > 0:

            elapsed_seconds += mido.tick2second(
                delta_ticks,
                mid.ticks_per_beat,
                current_tempo
            )

            previous_tick = absolute_tick

        # Note On
        if msg.type == "note_on" and msg.velocity > 0:

            notes.append({
                "time": elapsed_seconds,
                "note": msg.note,
                "velocity": msg.velocity
            })

    return notes


# ============================================================
# 動画情報取得
# ============================================================

def get_video_info(video_path):

    cap = cv2.VideoCapture(video_path)

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
# 動画処理
# ============================================================

def process_video(
    input_path,
    output_path,
    notes,
    mode,
    flash_duration
):

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

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # ノート時刻を取り出す
    note_times = np.array(
        sorted(
            [
                float(note["time"])
                for note in notes
            ]
        ),
        dtype=np.float64
    )

    # ========================================================
    # 一時動画
    # ========================================================

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

        cap.release()

        raise RuntimeError(
            "動画の書き込みを開始できませんでした。"
        )

    progress = st.progress(0)

    # ========================================================
    # フレームごとに処理
    # ========================================================

    for frame_number in range(
        frame_count
    ):

        ret, frame = cap.read()

        if not ret:
            break

        # このフレームの時間
        current_time = (
            frame_number / fps
        )

        should_invert = False

        # ====================================================
        # トグル
        # ====================================================

        if mode == "toggle":

            # この時刻までに鳴ったノート数
            note_count = np.searchsorted(
                note_times,
                current_time,
                side="right"
            )

            # 奇数回なら反転
            if note_count % 2 == 1:

                should_invert = True

        # ====================================================
        # フラッシュ
        # ====================================================

        elif mode == "flash":

            start_time = (
                current_time
                - flash_duration
            )

            left = np.searchsorted(
                note_times,
                start_time,
                side="left"
            )

            right = np.searchsorted(
                note_times,
                current_time,
                side="right"
            )

            if right > left:

                should_invert = True

        # ====================================================
        # 反転
        # ====================================================

        if should_invert:

            frame = cv2.bitwise_not(
                frame
            )

        # 書き込み
        writer.write(frame)

        # 進捗
        if frame_number % 10 == 0:

            progress.progress(
                min(
                    frame_number
                    / max(frame_count, 1),
                    1.0
                )
            )

    cap.release()
    writer.release()

    progress.progress(1.0)

    # ========================================================
    # FFmpegで元動画の音声を追加
    # ========================================================

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

    # 一時ファイル削除
    if os.path.exists(
        temp_video
    ):

        os.remove(
            temp_video
        )

    if result.returncode != 0:

        raise RuntimeError(
            "FFmpegでエラーが発生しました。\n\n"
            + result.stderr[-5000:]
        )


# ============================================================
# MIDIアップロード
# ============================================================

st.header("① MIDIファイル")

midi_file = st.file_uploader(
    "MIDIファイルを選択",
    type=[
        "mid",
        "midi"
    ]
)

mid = None
tracks = []
selected_notes = []


if midi_file:

    try:

        # 一時保存
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mid"
        ) as f:

            f.write(
                midi_file.getvalue()
            )

            midi_path = f.name

        # MIDI読み込み
        mid = mido.MidiFile(
            midi_path
        )

        tracks = get_tracks(
            mid
        )

        st.success(
            "MIDIを読み込みました！"
        )

        # ====================================================
        # トラック選択
        # ====================================================

        options = []

        for track in tracks:

            options.append(
                f"Track {track['index']} "
                f"— {track['name']} "
                f"— ノート数 {len(track['notes'])}"
            )

        selected_track_index = st.selectbox(
            "同期に使用するトラック",
            range(len(options)),
            format_func=lambda x: options[x]
        )

        # 選択トラックのノート
        selected_notes = (
            convert_track_notes_to_seconds(
                mid,
                selected_track_index
            )
        )

        st.write(
            f"🎵 ノート数："
            f"**{len(selected_notes)}**"
        )

        if selected_notes:

            st.write(
                "最初のノート："
                f" **{selected_notes[0]['time']:.6f} 秒**"
            )

            st.write(
                "最初の10個："
            )

            st.write(
                [
                    round(
                        n["time"],
                        4
                    )
                    for n in selected_notes[:10]
                ]
            )

        else:

            st.warning(
                "このトラックにはノートがありません。"
            )

    except Exception as e:

        st.error(
            "MIDI読み込みエラー："
            + str(e)
        )


# ============================================================
# 動画アップロード
# ============================================================

st.header("② 動画ファイル")

video_file = st.file_uploader(
    "動画ファイルを選択",
    type=[
        "mp4",
        "mov"
    ]
)

video_path = None
video_info = None


if video_file:

    try:

        extension = os.path.splitext(
            video_file.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as f:

            f.write(
                video_file.getvalue()
            )

            video_path = f.name

        video_info = get_video_info(
            video_path
        )

        st.success(
            "動画を読み込みました！"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "FPS",
            f"{video_info['fps']:.2f}"
        )

        col2.metric(
            "解像度",
            f"{video_info['width']} × "
            f"{video_info['height']}"
        )

        col3.metric(
            "長さ",
            f"{video_info['duration']:.2f} 秒"
        )

    except Exception as e:

        st.error(
            "動画読み込みエラー："
            + str(e)
        )


# ============================================================
# エフェクト設定
# ============================================================

st.header("③ エフェクト設定")

mode = st.radio(
    "反転方法",
    [
        "flash",
        "toggle"
    ],
    format_func=lambda x: {

        "flash":
            "⚡ フラッシュ "
            "（音が鳴った瞬間だけ反転）",

        "toggle":
            "🔄 トグル "
            "（音が鳴るたびON/OFF）"

    }[x]
)


if mode == "flash":

    flash_duration = st.slider(
        "反転時間（秒）",
        min_value=0.01,
        max_value=0.5,
        value=0.05,
        step=0.01
    )

    st.write(
        f"音が鳴った後 "
        f"**{flash_duration:.2f}秒** "
        f"反転します。"
    )

else:

    flash_duration = 0.05

    st.write(
        "音が鳴るたびに"
        "通常 → 反転 → 通常 → 反転"
        "と切り替わります。"
    )


# ============================================================
# 生成
# ============================================================

st.header("④ 動画生成")

if st.button(
    "🎬 YTPMV動画を生成する",
    type="primary",
    use_container_width=True
):

    # MIDIチェック
    if mid is None:

        st.error(
            "MIDIファイルをアップロードしてください。"
        )

        st.stop()

    # ノートチェック
    if not selected_notes:

        st.error(
            "ノートが入っているMIDIトラックを選択してください。"
        )

        st.stop()

    # 動画チェック
    if video_path is None:

        st.error(
            "動画ファイルをアップロードしてください。"
        )

        st.stop()

    # ========================================================
    # 生成
    # ========================================================

    output_path = os.path.join(
        tempfile.gettempdir(),
        "ytpmv_output.mp4"
    )

    try:

        with st.spinner(
            "動画を生成しています..."
        ):

            process_video(
                video_path,
                output_path,
                selected_notes,
                mode,
                flash_duration
            )

        st.success(
            "🎉 動画の生成が完了しました！"
        )

        # プレビュー
        st.video(
            output_path
        )

        # ダウンロード
        with open(
            output_path,
            "rb"
        ) as f:

            st.download_button(
                "⬇️ 完成した動画を保存",
                data=f,
                file_name="ytpmv_output.mp4",
                mime="video/mp4",
                use_container_width=True
            )

    except Exception as e:

        st.error(
            "動画生成中にエラーが発生しました。"
        )

        st.code(
            str(e)
        )
