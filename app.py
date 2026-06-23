import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import librosa
import soundfile as sf
import fingerprint as fp

# Set page config FIRST
st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide")

# Custom Dark Theme Styling with Outfit & JetBrains Mono Fonts
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0E1111;
        color: #E2F1F1;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Styles */
    .title-glowing {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 38px;
        background: linear-gradient(90deg, #00FFCC, #8A2BE2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 255, 204, 0.15);
        margin-bottom: 2px;
        margin-top: 5px;
    }
    .subtitle {
        color: #758A8A;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 2px;
        margin-bottom: 25px;
        text-transform: uppercase;
    }
    
    /* Premium Glassmorphic Cards */
    .glass-card {
        background: rgba(21, 25, 25, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(64, 224, 208, 0.15);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .glass-card:hover {
        border-color: rgba(64, 224, 208, 0.35);
        box-shadow: 0 12px 36px 0 rgba(64, 224, 208, 0.08);
        transform: translateY(-2px);
    }
    
    /* Metric Cards */
    .metric-box {
        background-color: #151919;
        border: 1px solid #222D2D;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-box:hover {
        border-color: #00FFCC;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 255, 204, 0.15);
    }
    .metric-title {
        font-size: 11px;
        color: #647C7C;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 20px;
        color: #00FFCC;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        margin: 6px 0px;
    }
    .metric-footer {
        font-size: 11px;
        color: #8FA3A3;
        font-weight: 500;
    }
    
    /* Custom Panels for matches */
    .match-found {
        background: linear-gradient(135deg, #112A25 0%, #0B1D19 100%);
        border: 1px solid #1E5C4E;
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(64, 224, 208, 0.1);
    }
    
    .success-panel {
        background: linear-gradient(135deg, #112A25 0%, #0B1D19 100%);
        border: 1px solid #00FF66;
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0, 255, 102, 0.1);
    }
    .warning-panel {
        background: linear-gradient(135deg, #2E2514 0%, #1A150B 100%);
        border: 1px solid #FFA500;
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(255, 165, 0, 0.1);
    }
    .danger-panel {
        background: linear-gradient(135deg, #2E1515 0%, #1F0D0D 100%);
        border: 1px solid #FF3333;
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(255, 51, 51, 0.1);
    }
    
    /* Custom Streamlit Elements overriding */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #151919;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #222D2D;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background-color: transparent;
        border-radius: 8px;
        color: #758A8A;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
        border: none;
        padding: 0px 20px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #00FFCC;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E4D43 !important;
        color: #00FFCC !important;
        box-shadow: 0 4px 10px rgba(64, 224, 208, 0.15);
    }
    
    /* Button Customization */
    div.stButton > button {
        background: linear-gradient(135deg, #1E4D43 0%, #112A25 100%);
        color: #00FFCC;
        border: 1px solid #1E4D43;
        border-radius: 8px;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        transition: all 0.3s ease;
        padding: 0.5rem 1.5rem;
        width: 100%;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #00FFCC 0%, #1E4D43 100%);
        color: #0E1111;
        border-color: #00FFCC;
        box-shadow: 0 0 12px rgba(0, 255, 204, 0.4);
        transform: translateY(-2px);
    }
    
    /* Expander override */
    div[data-testid="stExpander"] {
        background-color: #151919 !important;
        border: 1px solid #222D2D !important;
        border-radius: 12px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title Header
st.markdown("<h1 class='title-glowing'>EE200: Audio Fingerprinting</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Signals, Systems & Networks • Course Project</p>", unsafe_allow_html=True)

@st.cache_resource
def load_cached_database():
    if os.path.exists("database_index.pkl.gz"):
        import gzip
        with gzip.open("database_index.pkl.gz", "rb") as f:
            return pickle.load(f)
    elif os.path.exists("database_index.pkl"):
        with open("database_index.pkl", "rb") as f:
            return pickle.load(f)
    return None

db = load_cached_database()

if db is None:
    st.error("🚨 'database_index.pkl' or 'database_index.pkl.gz' not found! Please run indexing locally first via terminal commands.")
else:
    # Sidebar Info Panel
    st.sidebar.markdown("""
        <div style='background: linear-gradient(135deg, #151919 0%, #0F1212 100%); border: 1px solid #222D2D; padding:18px; border-radius:12px; margin-bottom:20px; box-shadow: 0 4px 6px rgba(0,0,0,0.15);'>
            <h3 style='color:#00FFCC; margin-top:0; font-family:sans-serif; font-weight:800; font-size:16px; letter-spacing:0.5px;'>PROJECT AUTHORS</h3>
            <p style='margin:6px 0px; font-size:14px; font-weight:600; color:#E2F1F1;'>👤 Arnab Patra</p>
            <p style='margin:0px 0px 10px 20px; font-size:12px; color:#8FA3A3;'>Roll No: 240186</p>
            <p style='margin:6px 0px; font-size:14px; font-weight:600; color:#E2F1F1;'>👤 Akshita</p>
            <p style='margin:0px 0px 10px 20px; font-size:12px; color:#8FA3A3;'>Roll No: 240090</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Calculate stats
    total_songs = len(db["songs"])
    total_hashes = len(db["index"])
    total_links = sum(len(v) for v in db["index"].values())
    
    st.sidebar.markdown(f"""
        <div style='background: linear-gradient(135deg, #151919 0%, #0F1212 100%); border: 1px solid #222D2D; padding:18px; border-radius:12px; box-shadow: 0 4px 6px rgba(0,0,0,0.15); margin-bottom:20px;'>
            <h3 style='color:#00FFCC; margin-top:0; font-family:sans-serif; font-weight:800; font-size:16px; letter-spacing:0.5px;'>DATABASE METRICS</h3>
            <p style='margin:6px 0px; font-size:13px; color:#E2F1F1;'>🗄️ <b>Songs Indexed:</b> {total_songs}</p>
            <p style='margin:6px 0px; font-size:13px; color:#E2F1F1;'>🔑 <b>Unique Keys:</b> {total_hashes:,}</p>
            <p style='margin:6px 0px; font-size:13px; color:#E2F1F1;'>⚓ <b>Total Hashes:</b> {total_links:,}</p>
            <p style='margin:6px 0px; font-size:13px; color:#E2F1F1;'>⚡ <b>System:</b> <span style="color:#00FF66; font-weight:700;">Active</span></p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar parameters tuner
    with st.sidebar.expander("⚙️ ALGORITHMIC SETTINGS", expanded=True):
        st.markdown("<p style='font-size:12px; color:#758A8A; margin-bottom: 12px;'>Tweak fingerprinting parameters dynamically:</p>", unsafe_allow_html=True)
        tuning_threshold = st.slider("Peak Threshold (dB)", -60, -10, -45, 5, help="Minimum decibel value for a spectrogram peak to be included.")
        tuning_neighborhood = st.slider("Neighborhood Size", 5, 30, 15, 2, help="Size of neighborhood filter. Smaller size yields denser peaks.")
        
        st.markdown("<p style='font-size:12px; color:#758A8A; margin-top:10px;'>Target Zone Bounds:</p>", unsafe_allow_html=True)
        dt_range = st.slider("Time Difference (dt)", 1, 100, (1, 50), help="Search window for coupling peaks in time frames.")
        df_range = st.slider("Freq Difference (df)", -50, 50, (-30, 30), help="Search window for coupling peaks in frequency bins.")
        
        tuning_dt = (dt_range[0], dt_range[1])
        tuning_df = (df_range[0], df_range[1])
        
        if tuning_neighborhood < 9 or tuning_threshold < -55:
            st.warning("⚠️ Warning: Denser peak settings might increase processing time.")

    # 4 Tab setup layout including the new Robustness Playground
    tab_lib, tab_id, tab_robust, tab_batch = st.tabs([
        "🎵 LIBRARY", "🔍 IDENTIFY", "🧪 ROBUSTNESS LAB", "📦 BATCH MATCH"
    ])

    # ==========================================
    # TAB 1: LIBRARY VIEW
    # ==========================================
    with tab_lib:
        st.markdown("<p style='color:#758A8A; font-size:14px; margin-bottom: 20px;'>Browse songs currently indexed in the database. Below are their sparse constellation thumbnails reconstructed from stored coordinates.</p>", unsafe_allow_html=True)
        
        songs_dict = db["songs"]
        song_names = sorted(list(songs_dict.keys()))
        
        # Display as a grid of 4 cards per row
        cols = st.columns(4)
        for idx, name in enumerate(song_names):
            with cols[idx % 4]:
                st.markdown(f"""
                <div style='background-color:#151919; border: 1px solid #222D2D; padding:15px; border-radius:10px; margin-bottom:15px; transition: all 0.3s;'>
                    <p style='margin:0px; font-weight:bold; color:#E2F1F1; font-size:14px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>{name}</p>
                    <p style='margin:0px; color:#647C7C; font-size:12px;'>{songs_dict[name]['hash_count']:,} hashes</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Tiny minimalist representation plot inside card
                fig, ax = plt.subplots(figsize=(3, 1))
                fig.patch.set_facecolor('#151919')
                ax.set_facecolor('#151919')
                peaks = songs_dict[name]["peaks"]
                if len(peaks) > 0:
                    t, f = zip(*peaks)
                    ax.scatter(t, f, color='#00FFCC', s=0.1, alpha=0.3)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()

    # ==========================================
    # TAB 2: IDENTIFY MODE
    # ==========================================
    with tab_id:
        st.subheader("Identify a clip")
        uploaded_file = st.file_uploader("Upload Audio Clip", type=["mp3", "wav", "m4a", "flac", "ogg"])
        
        # Playback uploaded audio
        if uploaded_file is not None:
            st.write("🔊 **Uploaded Clip Playback**")
            st.audio(uploaded_file)
            
        # Select test clip from library
        st.markdown("---")
        st.write("**OR SELECT A TEST CLIP FROM LIBRARY**")
        sample_selected = None
        
        sample_files = [f for f in os.listdir("query_clips") if f.endswith(('.mp3', '.wav'))] if os.path.exists("query_clips") else []
        
        for sf_name in sample_files[:5]:
            col_play, col_btn = st.columns([5, 1.2])
            with col_play:
                st.audio(os.path.join("query_clips", sf_name))
            with col_btn:
                st.write("")
                if st.button(f"⚡ Match {sf_name.split('_')[-1]}", key=sf_name):
                    sample_selected = os.path.join("query_clips", sf_name)

        target_path = None
        if uploaded_file is not None:
            target_path = f"temp_query_{uploaded_file.name}"
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        elif sample_selected is not None:
            target_path = sample_selected

        if target_path is not None:
            with st.spinner("Processing clip and calculating alignment..."):
                pred_song, votes, score_chart, offsets, stft_db, peaks, m = fp.match_clip(
                    target_path, db,
                    threshold_db=tuning_threshold,
                    neighborhood_size=tuning_neighborhood,
                    target_zone_dt=tuning_dt,
                    target_zone_df=tuning_df
                )
                
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
                <p style='margin:0px; color:#758A8A; font-size:12px; font-weight:bold; letter-spacing:1px;'>MATCH IDENTIFIED</p>
                <h2 style='margin:5px 0px; color:#00FFCC; font-weight: 800;'>🎵 {pred_song}</h2>
                <p style='margin:0px; font-size:13.5px; color:#A2B5B5;'>Clustering score: <b>{votes} votes</b> ({ratio}x higher than runner-up track)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Candidate Scores Table 
            st.write("### Candidate Scores (Top 5)")
            df_scores = pd.DataFrame(score_chart, columns=["Candidate Track", "Hits Score"])
            st.dataframe(df_scores.head(5), use_container_width=True)

            # --- VISUAL PIPELINE INTERMEDIATE STEPS ---
            st.markdown("---")
            st.markdown("### STEP 1 → FEATURE EXTRACTION\n**From spectrogram to constellation**")
            st.markdown("<p style='color:#758A8A; font-size:13px;'>The clip was converted into a time-frequency map (STFT); brighter channels mean louder coefficients at that instance. From that rich framework array, only the most prominent peaks were tracked. Discarding structural magnitude parameters yields extreme stability against noise and volume adjustments.</p>", unsafe_allow_html=True)
            
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                fig.patch.set_facecolor('#0E1111')
                ax.set_facecolor('#0E1111')
                ax.imshow(stft_db, origin='lower', aspect='auto', cmap='magma')
                ax.set_title("Query Track Spectrogram (dB)", color="white", fontsize=10)
                ax.tick_params(colors='gray', labelsize=8)
                st.pyplot(fig)
                plt.close()
            with p_col2:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                fig.patch.set_facecolor('#0E1111')
                ax.set_facecolor('#0E1111')
                if len(peaks) > 0:
                    t_p, f_p = zip(*peaks)
                    ax.scatter(t_p, f_p, color='#00FFCC', s=2)
                ax.set_title("Extracted Landmark Constellation Map", color="white", fontsize=10)
                ax.set_xlim(0, stft_db.shape[1])
                ax.set_ylim(0, stft_db.shape[0])
                ax.tick_params(colors='gray', labelsize=8)
                st.pyplot(fig)
                plt.close()

            st.markdown("### STEP 2 → DATABASE SEARCH\n**Constellation Peak Alignment Overlay**")
            st.markdown(f"<p style='color:#758A8A; font-size:13px;'>The fingerprint hashes were matched against the index. Below is the constellation layout of <b>{pred_song}</b> (in gray) reconstructed from stored coordinates. The identified query window is highlighted, and the <b>query's peaks (shifted by the calculated offset) are overlaid in neon green</b> to show exact alignment.</p>", unsafe_allow_html=True)
            
            if pred_song in db["songs"]:
                fig, ax = plt.subplots(figsize=(12, 4))
                fig.patch.set_facecolor('#0E1111')
                ax.set_facecolor('#0E1111')
                ref_peaks = db["songs"][pred_song]["peaks"]
                if len(ref_peaks) > 0:
                    rt, rf = zip(*ref_peaks)
                    ax.scatter(rt, rf, color='#556666', s=1.5, alpha=0.4, label="Reference Peaks")
                
                # Overlay query peaks shifted by best_offset
                offset_center = m["best_offset"]
                if len(peaks) > 0:
                    qt, qf = zip(*peaks)
                    qt_shifted = [t + offset_center for t in qt]
                    ax.scatter(qt_shifted, qf, color='#00FFCC', s=3.5, alpha=0.9, label="Aligned Query Peaks")
                
                # Highlight search window
                ax.axvspan(max(0, offset_center), offset_center + stft_db.shape[1], color='#1E4D43', alpha=0.25, label="Matched Window")
                
                ax.set_title(f"Constellation Synchronization Map ({pred_song})", color="white", fontsize=11, fontweight='bold')
                ax.set_xlabel("Time Frames", color="#758A8A", fontsize=9)
                ax.set_ylabel("Frequency Bins", color="#758A8A", fontsize=9)
                ax.tick_params(colors='gray', labelsize=8)
                ax.legend(facecolor='#151919', edgecolor='#222D2D', labelcolor='white', loc='upper right')
                ax.grid(color='#222D2D', linestyle=':', linewidth=0.5)
                st.pyplot(fig)
                plt.close()

            st.markdown("### STEP 3 → THE PROOF\n**The alignment spike**")
            st.markdown("<p style='color:#758A8A; font-size:13px;'>Every true matched pair casts a vote for a relative time offset delta. Coincidental collisions scatter randomly, shaping a flat low background floor. Genuine alignments converge on a single exact frame calculation offset, forming a massive undeniable spike.</p>", unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(12, 3.5))
            fig.patch.set_facecolor('#0E1111')
            ax.set_facecolor('#0E1111')
            if len(offsets) > 0:
                counts, bins, _ = ax.hist(offsets, bins=100, color='#FFD700', edgecolor='none', alpha=0.9)
                # Render chance noise floor line estimation parameter
                noise_floor = np.mean(counts) + 2
                ax.axhline(noise_floor, color='red', linestyle='--', alpha=0.6, label="Chance matches (noise floor)")
            ax.set_title(r"Time Offset Histogram Distribution ($\Delta T$ Alignment)", color="white", fontsize=10)
            ax.tick_params(colors='gray', labelsize=8)
            ax.grid(color='#222D2D', linestyle='-', linewidth=0.5)
            st.pyplot(fig)
            plt.close()
            
            if uploaded_file is not None and os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except:
                    pass

    # ==========================================
    # TAB 3: ROBUSTNESS PLAYGROUND (SIMULATION LAB)
    # ==========================================
    with tab_robust:
        st.subheader("🧪 Robustness Simulation Lab")
        st.markdown("<p style='color:#758A8A; font-size:13px;'>Select a song from your indexed library and slice a custom segment. Then, apply environmental distortions (additive Gaussian noise, pitch transposition, or time-stretching) to simulate real-world recordings and test search limits.</p>", unsafe_allow_html=True)
        
        col_c1, col_c2 = st.columns([2, 3])
        
        with col_c1:
            st.markdown("#### 🛠️ Configure Simulation")
            selected_song = st.selectbox("Select Target Library Song", song_names)
            
            # Find the actual file in song_library
            song_path = None
            if os.path.exists("song_library"):
                for f in os.listdir("song_library"):
                    if os.path.splitext(f)[0] == selected_song:
                        song_path = os.path.join("song_library", f)
                        break
            
            # Duration and offset selection
            clip_dur = st.slider("Clip Duration (seconds)", 3, 20, 10, 1)
            start_offset = st.slider("Start Offset in Track (seconds)", 0, 120, 10, 5)
            
            st.markdown("#### ⚡ Add Distortions (Environmental)")
            noise_level = st.slider("Gaussian Noise Level (relative std)", 0.0, 1.0, 0.0, 0.05, help="Simulate a noisy café or restaurant background noise.")
            pitch_shift = st.slider("Pitch Shift (semitones)", -3.0, 3.0, 0.0, 0.2, help="Transpose the notes of the clip. Absolute bin matching fails on shifts.")
            time_stretch = st.slider("Time Stretch (playback rate)", 0.8, 1.2, 1.0, 0.05, help="Speed up (>1.0) or slow down (<1.0) the playback speed without changing pitch.")
            
            run_sim = st.button("🚀 Synthesize & Identify")
            
        with col_c2:
            st.markdown("#### 📺 Simulation Output & Visualizations")
            
            if run_sim:
                if song_path is None or not os.path.exists("song_library"):
                    st.warning("⚠️ **Source Audio Files Missing**: The `song_library/` directory is not available on this server host (it is gitignored to keep repository sizes small). Running audio mutations requires local access to original audio tracks. Please run the application locally or upload any query clip directly in the **🔍 IDENTIFY** tab!")
                else:
                    with st.spinner("Synthesizing audio snippet and running distortion algorithms..."):
                        # 1. Load the clean snippet
                        try:
                            y_clean, sr = librosa.load(song_path, sr=22050, offset=start_offset, duration=clip_dur)
                        except Exception as e:
                            st.error(f"Error loading target song: {e}")
                            y_clean = None
                        
                        if y_clean is not None:
                            # 2. Apply Pitch Shift
                            if pitch_shift != 0.0:
                                y_mutated = librosa.effects.pitch_shift(y_clean, sr=sr, n_steps=pitch_shift)
                            else:
                                y_mutated = y_clean.copy()
                            
                            # 3. Apply Time Stretch
                            if time_stretch != 1.0:
                                y_mutated = librosa.effects.time_stretch(y_mutated, rate=time_stretch)
                            
                            # 4. Apply Gaussian Noise
                            if noise_level > 0.0:
                                noise = np.random.normal(0, noise_level, len(y_mutated))
                                y_mutated = y_mutated + noise
                            
                            # 5. Write to temporary wav file
                            temp_robust_path = "temp_robust_query.wav"
                            sf.write(temp_robust_path, y_mutated, sr)
                            
                            # 6. Playback Mutated Clip
                            st.write("🔊 **Mutated Audio Playback**")
                            st.audio(temp_robust_path)
                            
                            # 7. Run Match clip
                            pred_song, votes, score_chart, offsets, stft_mut, peaks_mut, m_mut = fp.match_clip(
                                temp_robust_path, db,
                                threshold_db=tuning_threshold,
                                neighborhood_size=tuning_neighborhood,
                                target_zone_dt=tuning_dt,
                                target_zone_df=tuning_df
                            )
                            
                            # 8. Success / Failure / Mismatch Banner
                            if pred_song == selected_song:
                                st.markdown(f"""
                                <div class='success-panel'>
                                    <h4 style='color: #00FF66; margin: 0; font-weight:800;'>✅ SUCCESSFUL IDENTIFICATION</h4>
                                    <p style='color: #E2F1F1; margin: 6px 0 0 0; font-size:14px;'>The system successfully matched the distorted clip to <b>{pred_song}</b> with <b>{votes}</b> alignment votes!</p>
                                </div>
                                """, unsafe_allow_html=True)
                            elif pred_song == "Unknown / No Match":
                                st.markdown(f"""
                                <div class='warning-panel'>
                                    <h4 style='color: #FFA500; margin: 0; font-weight:800;'>⚠️ RECOGNITION FAILURE (NO MATCH)</h4>
                                    <p style='color: #E2F1F1; margin: 6px 0 0 0; font-size:14px;'>The applied distortions were too severe. The clip did not cross the matching threshold.</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class='danger-panel'>
                                    <h4 style='color: #FF3333; margin: 0; font-weight:800;'>❌ MISMATCHED TRACK</h4>
                                    <p style='color: #E2F1F1; margin: 6px 0 0 0; font-size:14px;'>The distortion caused a false match! The system predicted <b>{pred_song}</b> instead of <b>{selected_song}</b>.</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # 9. Compare Clean vs Distorted Spectrograms
                            st.write("---")
                            st.write("**STFT Spectrogram Comparison (Clean vs. Mutated)**")
                            
                            # Generate clean spectrogram & peaks
                            stft_clean = fp.compute_spectrogram(y_clean, sr)
                            peaks_clean = fp.get_constellation_map(stft_clean, threshold_db=tuning_threshold, neighborhood_size=tuning_neighborhood)
                            
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
                            fig.patch.set_facecolor('#0E1111')
                            ax1.set_facecolor('#0E1111')
                            ax2.set_facecolor('#0E1111')
                            
                            # Clean Spectrogram
                            ax1.imshow(stft_clean, origin='lower', aspect='auto', cmap='magma', alpha=0.6)
                            if len(peaks_clean) > 0:
                                tc, fc = zip(*peaks_clean)
                                ax1.scatter(tc, fc, color='#00FFCC', s=2.5, label="Clean Peaks")
                            ax1.set_title("Clean Reference Snippet Peaks", color="white", fontsize=9, fontweight='bold')
                            ax1.tick_params(colors='gray', labelsize=8)
                            ax1.set_xlabel("Time Frames", color="#758A8A", fontsize=8)
                            ax1.set_ylabel("Frequency Bins", color="#758A8A", fontsize=8)
                            
                            # Distorted Spectrogram
                            ax2.imshow(stft_mut, origin='lower', aspect='auto', cmap='magma', alpha=0.6)
                            if len(peaks_mut) > 0:
                                tm, fm = zip(*peaks_mut)
                                ax2.scatter(tm, fm, color='red', s=2.5, label="Mutated Peaks")
                            ax2.set_title("Mutated Query Snippet Peaks", color="white", fontsize=9, fontweight='bold')
                            ax2.tick_params(colors='gray', labelsize=8)
                            ax2.set_xlabel("Time Frames", color="#758A8A", fontsize=8)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
            else:
                st.info("Configure the parameters in the left panel and click 'Synthesize & Identify' to run the robustness simulation.")

    # ==========================================
    # TAB 4: BATCH PROCESSING MODE
    # ==========================================
    with tab_batch:
        st.subheader("Identify many clips at once")
        st.markdown("<p style='color:#758A8A; font-size:13px;'>Upload a set of query clips. Each is identified against the currently indexed library, and the results are written to a standardised <b>results.csv</b> with columns <code>filename</code>, <code>prediction</code> (without extension).</p>", unsafe_allow_html=True)
        
        batch_files = st.file_uploader("Upload multiple query audio files", type=["mp3", "wav"], accept_multiple_files=True, key="batch")
        
        if st.button("Run Batch Recognition") and batch_files:
            batch_results = []
            progress_bar = st.progress(0)
            
            for idx, file in enumerate(batch_files):
                b_path = f"temp_b_{idx}_{file.name}"
                with open(b_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Execute match
                pred_song, votes, _, _, _, _, _ = fp.match_clip(
                    b_path, db,
                    threshold_db=tuning_threshold,
                    neighborhood_size=tuning_neighborhood,
                    target_zone_dt=tuning_dt,
                    target_zone_df=tuning_df
                )
                
                # Format predictions correctly
                final_label = pred_song if pred_song != "Unknown / No Match" else "none"
                
                batch_results.append({
                    "filename": file.name,
                    "prediction": final_label
                })
                
                if os.path.exists(b_path):
                    try:
                        os.remove(b_path)
                    except:
                        pass
                progress_bar.progress((idx + 1) / len(batch_files))
            
            df_batch = pd.DataFrame(batch_results)
            
            # Show summary stats
            matched_count = sum(1 for r in batch_results if r["prediction"] != "none")
            unmatched_count = len(batch_results) - matched_count
            success_rate = (matched_count / len(batch_results)) * 100
            
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.markdown(f"<div class='metric-box'><div class='metric-title'>📈 MATCHED</div><div class='metric-value'>{matched_count}</div><div class='metric-footer'>Tracks identified</div></div>", unsafe_allow_html=True)
            with col_b2:
                st.markdown(f"<div class='metric-box'><div class='metric-title'>📉 UNMATCHED</div><div class='metric-value'>{unmatched_count}</div><div class='metric-footer'>Unknown or low votes</div></div>", unsafe_allow_html=True)
            with col_b3:
                st.markdown(f"<div class='metric-box'><div class='metric-title'>🎯 SUCCESS RATE</div><div class='metric-value'>{success_rate:.1f}%</div><div class='metric-footer'>Overall confidence</div></div>", unsafe_allow_html=True)
            
            st.write("### Batch Results Table")
            st.dataframe(df_batch, use_container_width=True)
            
            csv_data = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download results.csv",
                data=csv_data,
                file_name="results.csv",
                mime="text/csv"
            )