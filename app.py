import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import fingerprint as fp

st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide")

# Custom Dark Theme Styling to match the demo video exactly
st.markdown("""
    <style>
    body, .main, .block-container { background-color: #111414; color: #E0E0E0; }
    h1 { color: #E2F1F1; font-family: monospace; font-weight: bold; margin-bottom:0px; }
    .subtitle { color: #758A8A; font-size: 13px; font-weight: bold; letter-spacing: 1px; margin-bottom: 20px;}
    .metric-box { background-color: #161B1B; border: 1px solid #232D2D; border-radius: 8px; padding: 10px; text-align: center; }
    .metric-title { font-size: 11px; color: #647C7C; font-weight: bold; letter-spacing: 0.5px; text-transform: uppercase;}
    .metric-value { font-size: 18px; color: #40E0D0; font-family: monospace; font-weight: bold; margin: 4px 0px;}
    .metric-footer { font-size: 11px; color: #8FA3A3; }
    .match-found { background-color: #122522; border: 1px solid #1E4D43; padding: 15px; border-radius: 6px; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>EE200: Audio Fingerprinting</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>SIGNALS, SYSTEMS & NETWORKS • PROJECT DEMO</p>", unsafe_allow_html=True)

@st.cache_resource
def load_cached_database():
    if os.path.exists("database_index.pkl"):
        with open("database_index.pkl", "rb") as f:
            return pickle.load(f)
    return None

db = load_cached_database()

if db is None:
    st.error("🚨 'database_index.pkl' not found! Please run indexing locally first via terminal commands.")
else:
    # 3 Tab setup layout from the demo clip
    tab_lib, tab_id, tab_batch = st.tabs(["LIBRARY", "IDENTIFY", "BATCH"])

    # ==========================================
    # TAB 1: LIBRARY VIEW
    # ==========================================
    with tab_lib:
        st.markdown("<p style='color:#758A8A; font-size:13px;'>Index a library of songs as spectrogram fingerprints, then identify any short clip against it.</p>", unsafe_allow_html=True)
        
        songs_dict = db["songs"]
        song_names = list(songs_dict.keys())
        
        # Display as a grid of 4 cards per row
        cols = st.columns(4)
        for idx, name in enumerate(song_names):
            with cols[idx % 4]:
                st.markdown(f"""
                <div style='background-color:#161B1B; border: 1px solid #232D2D; padding:15px; border-radius:8px; margin-bottom:15px;'>
                    <p style='margin:0px; font-weight:bold; color:#E2F1F1; font-size:14px;'>{name}</p>
                    <p style='margin:0px; color:#647C7C; font-size:12px;'>{songs_dict[name]['hash_count']:,} hashes</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Tiny minimalist representation plot inside card
                fig, ax = plt.subplots(figsize=(3, 1))
                fig.patch.set_facecolor('#161B1B')
                ax.set_facecolor('#161B1B')
                peaks = songs_dict[name]["peaks"]
                if len(peaks) > 0:
                    t, f = zip(*peaks)
                    ax.scatter(t, f, color='#40E0D0', s=0.1, alpha=0.4)
                ax.axis('off')
                st.pyplot(fig)

    # ==========================================
    # TAB 2: IDENTIFY MODE
    # ==========================================
    with tab_id:
        st.subheader("Identify a clip")
        uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a", "flac", "ogg"])
        
        # Hardcoded sample clips option mock matches to match video try choices
        st.markdown("---")
        st.write("**OR TRY A SAMPLE**")
        sample_selected = None
        
        sample_files = [f for f in os.listdir("query_clips") if f.endswith(('.mp3', '.wav'))] if os.path.exists("query_clips") else []
        
        for sf in sample_files[:5]:
            col_play, col_btn = st.columns([5, 1])
            with col_play:
                st.audio(os.path.join("query_clips", sf))
            with col_btn:
                if st.button(f"Try {sf}", key=sf):
                    sample_selected = os.path.join("query_clips", sf)

        target_path = None
        if uploaded_file is not None:
            target_path = f"temp_query_{uploaded_file.name}"
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        elif sample_selected is not None:
            target_path = sample_selected

        if target_path is not None:
            # Match
            pred_song, votes, score_chart, offsets, stft_db, peaks, m = fp.match_clip(target_path, db)
            
            # Display real-time speed metric boxes
            tot_time = m['spectrogram_time'] + m['constellation_time'] + m['hashes_time'] + m['db_time'] + m['scoring_time']
            
            met_col1, met_col2, met_col3, met_col4, met_col5, met_col6 = st.columns(6)
            with met_col1:
                st.markdown(f"<div class='metric-box'><div class='metric-title'>⏱️ Spectrogram</div><div class='metric-value'>{m['spectrogram_time']} ms</div><div class='metric-footer'>{m['spectrogram_shape']}</div></div>", unsafe_allow_html=True)
            with met_col2:
                st.markdown(f"<div class='metric-box'><div class='metric-title'>✨ Constellation</div><div class='metric-value'>{m['constellation_time']} ms</div><div class='metric-footer'>{m['peak_count']} peaks</div></div>", unsafe_allow_html=True)
            with met_col3:
                st.markdown(f"<div class='metric-box'><div class='metric-title'>⚓ Hashing</div><div class='metric-value'>{m['hashes_time']} ms</div><div class='metric-footer'>{m['hash_count']} pairs</div></div>", unsafe_allow_html=True)
            with met_col4:
                st.markdown(f"<div class='metric-box'><div class='metric-title'>🔍 DB Lookup</div><div class='metric-value'>{m['db_time']} ms</div><div class='metric-footer'>{m['tracks_checked']} tracks</div></div>", unsafe_allow_html=True)
            with met_col5:
                st.markdown(f"<div class='metric-box'><div class='metric-title'>📊 Scoring</div><div class='metric-value'>{m['scoring_time']} ms</div><div class='metric-footer'>offset {m['best_offset']}</div></div>", unsafe_allow_html=True)
            with met_col6:
                st.markdown(f"<div class='metric-box'><div class='metric-title'>⚡ TOTAL TIME</div><div class='metric-value' style='color:#FFD700'>{tot_time} ms</div><div class='metric-footer'>Completed</div></div>", unsafe_allow_html=True)
            
            # Winner Announcement banner 
            runner_up_votes = score_chart[1][1] if len(score_chart) > 1 else 1
            ratio = round(votes / max(runner_up_votes, 1), 1)
            
            st.markdown(f"""
            <div class='match-found'>
                <p style='margin:0px; color:#758A8A; font-size:12px; font-weight:bold; letter-spacing:0.5px;'>MATCH FOUND</p>
                <h2 style='margin:5px 0px; color:#40E0D0;'>{pred_song}</h2>
                <p style='margin:0px; font-size:13px; color:#A2B5B5;'>Cluster score: <b>{votes}</b> ({ratio}x higher than runner up)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Candidate Scores Table 
            st.write("### Candidate Scores")
            df_scores = pd.DataFrame(score_chart, columns=["Candidate Track", "Hits Score"])
            st.dataframe(df_results := df_scores.head(5), use_container_width=True)

            # --- VISUAL PIPELINE INTERMEDIATE STEPS ---
            st.markdown("---")
            st.markdown("### STEP 1 → FEATURE EXTRACTION\n**From spectrogram to constellation**")
            st.markdown("<p style='color:#758A8A; font-size:13px;'>The clip was converted into a time-frequency map (STFT); brighter channels mean louder coefficients at that instance. From that rich framework array, only the most prominent peaks were tracked. Discarding structural magnitude parameters yields extreme stability against noise and volume adjustments.</p>", unsafe_allow_html=True)
            
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                fig.patch.set_facecolor('#111414')
                ax.set_facecolor('#111414')
                ax.imshow(stft_db, origin='lower', aspect='auto', cmap='magma')
                ax.set_title("Query Track Spectrogram (dB)", color="white", fontsize=10)
                ax.tick_params(colors='gray', labelsize=8)
                st.pyplot(fig)
            with p_col2:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                fig.patch.set_facecolor('#111414')
                ax.set_facecolor('#111414')
                if len(peaks) > 0:
                    t_p, f_p = zip(*peaks)
                    ax.scatter(t_p, f_p, color='#40E0D0', s=2)
                ax.set_title("Extracted Landmark Constellation Map", color="white", fontsize=10)
                ax.set_xlim(0, stft_db.shape[1])
                ax.set_ylim(0, stft_db.shape[0])
                ax.tick_params(colors='gray', labelsize=8)
                st.pyplot(fig)

            st.markdown("### STEP 2 → DATABASE SEARCH\n**Where in the song?**")
            st.markdown(f"<p style='color:#758A8A; font-size:13px;'>The fingerprint hashes were matched against the index. Below is the structural layout of <b>{pred_song}</b> reconstructed via stored coordinates, where the highlighted window showcases where your query syncs perfectly.</p>", unsafe_allow_html=True)
            
            if pred_song in db["songs"]:
                fig, ax = plt.subplots(figsize=(12, 3))
                fig.patch.set_facecolor('#111414')
                ax.set_facecolor('#111414')
                ref_peaks = db["songs"][pred_song]["peaks"]
                if len(ref_peaks) > 0:
                    rt, rf = zip(*ref_peaks)
                    ax.scatter(rt, rf, color='gray', s=0.8, alpha=0.4)
                
                # Draw high visible query alignment focus window boundaries
                offset_center = m["best_offset"]
                ax.axvspan(max(0, offset_center), offset_center + stft_db.shape[1], color='#1E4D43', alpha=0.4, label="Identified Match Segment")
                ax.set_title(f"Full Reference Target Framework Time-Sync Mapping Location: {pred_song}", color="white", fontsize=10)
                ax.axis('off')
                st.pyplot(fig)

            st.markdown("### STEP 3 → THE PROOF\n**The alignment spike**")
            st.markdown("<p style='color:#758A8A; font-size:13px;'>Every true matched pair casts a vote for a relative time offset delta. Coincidental collisions scatter randomly, shaping a flat low background floor. Genuine alignments converge on a single exact frame calculation offset, forming a massive undeniable spike.</p>", unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(12, 3.5))
            fig.patch.set_facecolor('#111414')
            ax.set_facecolor('#111414')
            if len(offsets) > 0:
                counts, bins, _ = ax.hist(offsets, bins=100, color='#FFD700', edgecolor='none', alpha=0.9)
                # Render chance noise floor line estimation parameter
                noise_floor = np.mean(counts) + 2
                ax.axhline(noise_floor, color='red', linestyle='--', alpha=0.6, label="Chance matches (noise floor)")
            ax.set_title("Time Offset Histogram Distribution ($\Delta T$ Alignment)", color="white", fontsize=10)
            ax.tick_params(colors='gray', labelsize=8)
            ax.grid(color='#232D2D', linestyle='-', linewidth=0.5)
            st.pyplot(fig)
            
            if uploaded_file is not None and os.path.exists(target_path):
                os.remove(target_path)

    # ==========================================
    # TAB 3: BATCH PROCESSING MODE
    # ==========================================
    with tab_batch:
        st.subheader("Identify many clips at once")
        st.markdown("<p style='color:#758A8A; font-size:13px;'>Upload a set of query clips. Each is identified against the currently indexed library, and the results are written to a standardised <b>results.csv</b> with columns <code>filename</code>, <code>prediction</code> (without extension).</p>", unsafe_allow_html=True)
        
        batch_files = st.file_uploader("Upload an array of files", type=["mp3", "wav"], accept_multiple_files=True, key="batch")
        
        if st.button("Run batch") and batch_files:
            batch_results = []
            progress_bar = st.progress(0)
            
            for idx, file in enumerate(batch_files):
                b_path = f"temp_b_{idx}_{file.name}"
                with open(b_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Execute match
                pred_song, votes, _, _, _, _, _ = fp.match_clip(b_path, db)
                
                # Check evaluation confidence cutoff threshold criteria (e.g. at least 5 votes required)
                final_label = pred_song if votes >= 5 else "none"
                
                batch_results.append({
                    "filename": file.name,
                    "prediction": final_label
                })
                
                if os.path.exists(b_path):
                    os.remove(b_path)
                progress_bar.progress((idx + 1) / len(batch_files))
            
            df_batch = pd.DataFrame(batch_results)
            st.write("### Results")
            st.dataframe(df_batch, use_container_width=True)
            
            csv_data = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download results.csv",
                data=csv_data,
                file_name="results.csv",
                mime="text/csv"
            )