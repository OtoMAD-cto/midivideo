import tempfile
import os
import streamlit as st
import mido
import numpy as np
import cv2

st.title("MIDI to Video Generator")
st.write("MIDIファイルをアップロードして、簡易的な動画に変換します。")

uploaded_file = st.file_uploader("MIDIファイルをアップロード (.mid)", type=["mid", "midi"])

if uploaded_file is not None:
    st.success("ファイルのアップロードに成功しました！")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_midi_path = tmp_file.name

    try:
        mid = mido.MidiFile(tmp_midi_path)
        st.write(f"トラック数: {len(mid.tracks)}")

        if st.button("動画に変換する"):
            with st.spinner("動画を作成中...（しばらくお待ちください）"):
                
                output_video_path = "output.mp4"
                fps = 30
                width, height = 640, 360
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                
                for i in range(fps * 3): # 3秒間のダミー動画
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(frame, f"Frame: {i}", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    out.write(frame)
                
                out.release()
                
                st.success("動画の変換が完了しました！")
                with open(output_video_path, "rb") as file:
                    st.download_button(
                        label="動画をダウンロード",
                        data=file,
                        file_name="midi_output.mp4",
                        mime="video/mp4"
                    )

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        
    finally:
        if os.path.exists(tmp_midi_path):
            os.remove(tmp_midi_path)
