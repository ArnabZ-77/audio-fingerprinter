import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import fingerprint as fp
import pickle
import gzip

def match_signal(y, sr, database):
    """Utility to match a pre-loaded numpy signal array against the database index (no disk write)."""
    stft_db = fp.compute_spectrogram(y, sr)
    peaks = fp.get_constellation_map(stft_db)
    query_hashes = fp.generate_hashes(peaks)
    
    matches = {}
    for hash_key, q_t1_list in query_hashes.items():
        if hash_key in database["index"]:
            for song_ref, db_t1 in database["index"][hash_key]:
                if isinstance(song_ref, int):
                    song_name = database.get("id_to_song", {}).get(song_ref, str(song_ref))
                else:
                    song_name = song_ref
                for q_t1 in q_t1_list:
                    offset = db_t1 - q_t1
                    if song_name not in matches:
                        matches[song_name] = []
                    matches[song_name].append(offset)
    
    best_song = "Unknown / No Match"
    highest_vote_count = 0
    for song_name, offsets in matches.items():
        if len(offsets) == 0:
            continue
        counts, bins = np.histogram(offsets, bins=np.arange(min(offsets)-1, max(offsets)+2))
        max_votes = np.max(counts)
        if max_votes > highest_vote_count:
            highest_vote_count = max_votes
            best_song = song_name
    return best_song, highest_vote_count

def run_analytical_experiments():
    os.makedirs("report_figures", exist_ok=True)
    songs = [f for f in os.listdir("song_library") if f.endswith(('.mp3', '.wav'))]
    
    if not songs:
        print("Please place audio tracks in 'song_library' folder before compiling metrics.")
        return
        
    target_track = os.path.join("song_library", songs[0])
    correct_song_name = os.path.splitext(songs[0])[0]
    print(f"Running experiments using baseline reference audio track: {target_track}")
    y, sr = fp.load_audio(target_track)
    
    # Load database index for sweep experiments
    db_path_gz = "database_index.pkl.gz"
    db_path = "database_index.pkl"
    if os.path.exists(db_path_gz):
        print(f"Loading database index from '{db_path_gz}'...")
        with gzip.open(db_path_gz, "rb") as f:
            db = pickle.load(f)
    elif os.path.exists(db_path):
        print(f"Loading database index from '{db_path}'...")
        with open(db_path, "rb") as f:
            db = pickle.load(f)
    else:
        print("No database index found. Building index first...")
        fp.build_database("song_library")
        if os.path.exists(db_path_gz):
            with gzip.open(db_path_gz, "rb") as f:
                db = pickle.load(f)
        else:
            with open(db_path, "rb") as f:
                db = pickle.load(f)
            
    # 1. Fourier Domain Analysis Comparison (DFT vs. STFT Spectrogram)
    print("Generating Experiment 1: DFT vs. Spectrogram...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    dft = np.abs(np.fft.fft(y))[:len(y)//2]
    freqs = np.fft.fftfreq(len(y), 1/sr)[:len(y)//2]
    ax1.plot(freqs, dft, color='purple')
    ax1.set_title("Standard Global DFT Spectrum (No Time Context)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude")
    
    stft_db = fp.compute_spectrogram(y, sr)
    im = ax2.imshow(stft_db, origin='lower', aspect='auto', cmap='magma')
    ax2.set_title("STFT Spectrogram Matrix Representation (Time + Frequency)")
    ax2.set_xlabel("Time Frames")
    ax2.set_ylabel("Frequency Bins")
    plt.tight_layout()
    plt.savefig("report_figures/exp1_dft_vs_spectrogram.png")
    plt.close()

    # 2. Structural Window Slices Resolution Contrast (Short vs. Long Window sizes)
    print("Generating Experiment 2: Window Length Tradeoffs...")
    stft_short = fp.compute_spectrogram(y, sr, n_fft=256, hop_length=64)
    stft_long = fp.compute_spectrogram(y, sr, n_fft=4096, hop_length=1024)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.imshow(stft_short, origin='lower', aspect='auto', cmap='viridis')
    ax1.set_title("Short Window (n_fft=256): High Time / Blurry Frequency Resolution")
    ax2.imshow(stft_long, origin='lower', aspect='auto', cmap='viridis')
    ax2.set_title("Long Window (n_fft=4096): High Frequency / Blurry Time Resolution")
    plt.tight_layout()
    plt.savefig("report_figures/exp2_window_lengths.png")
    plt.close()

    # 3. Noise Contamination Analysis & Noise Sweep
    print("Generating Experiment 3: Environmental Noise Robustness & Sweep...")
    y_snippet = y[:15 * sr]
    stft_snippet = fp.compute_spectrogram(y_snippet, sr)
    noise = np.random.normal(0, 0.15, len(y_snippet))
    noisy_y_snippet = y_snippet + noise
    stft_noisy = fp.compute_spectrogram(noisy_y_snippet, sr)
    peaks_clean = fp.get_constellation_map(stft_snippet)
    peaks_noisy = fp.get_constellation_map(stft_noisy)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.imshow(stft_snippet, origin='lower', aspect='auto', cmap='gray', alpha=0.3)
    t_c, f_c = zip(*peaks_clean) if peaks_clean else ([], [])
    ax1.scatter(t_c, f_c, color='cyan', s=2)
    ax1.set_title(f"Clean Constellation Map ({len(peaks_clean)} Points)")
    
    ax2.imshow(stft_noisy, origin='lower', aspect='auto', cmap='gray', alpha=0.3)
    t_n, f_n = zip(*peaks_noisy) if peaks_noisy else ([], [])
    ax2.scatter(t_n, f_n, color='red', s=2)
    ax2.set_title(f"Noise Contaminated Constellation Map ({len(peaks_noisy)} Points)")
    plt.tight_layout()
    plt.savefig("report_figures/exp3_noise_robustness.png")
    plt.close()

    # Noise level sweep
    noise_levels = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    sweep_votes = []
    correct_matches = []
    y_test_clip = y[10*sr : 25*sr] # 15-second test query
    
    for level in noise_levels:
        noise = np.random.normal(0, level, len(y_test_clip))
        noisy_sig = y_test_clip + noise
        pred, votes = match_signal(noisy_sig, sr, db)
        is_correct = (pred == correct_song_name)
        sweep_votes.append(votes if is_correct else 0)
        correct_matches.append(is_correct)
        print(f"  Noise std={level:.2f}: Predicted='{pred}', Votes={votes} (Correct: {is_correct})")
        
    plt.figure(figsize=(8, 4.5))
    plt.plot(noise_levels, sweep_votes, marker='o', linewidth=2, color='teal', label='Correct Match Votes')
    # Draw cutoff line where classification breaks down
    fail_idx = -1
    for i, match in enumerate(correct_matches):
        if not match:
            fail_idx = i
            break
            
    if fail_idx != -1:
        plt.axvline(noise_levels[fail_idx], color='red', linestyle='--', label=f'Recognition Failure Threshold (>{noise_levels[fail_idx-1]:.2f})')
    plt.title("Noise Robustness Sweep: Matching Votes vs. Noise Level")
    plt.xlabel("Gaussian Noise Standard Deviation (Relative to Signal)")
    plt.ylabel("Clustering Spike Votes (Hits Score)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("report_figures/exp3_noise_sweep.png")
    plt.close()

    # 4. Single Peaks vs Paired Hashes Matching Comparison
    print("Generating Experiment 4: Single Peaks vs Paired Hashes Matching...")
    peaks = fp.get_constellation_map(stft_db)
    hashes_paired = fp.generate_hashes(peaks)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Single peaks histogram
    if peaks:
        t_peaks, f_peaks = zip(*peaks)
        ax1.hist(t_peaks, bins=50, color='orange', alpha=0.7)
        ax1.set_title("Single Peaks Approach: Time Distribution (Sparse)")
        ax1.set_xlabel("Time Frame")
        ax1.set_ylabel("Count")
        ax1.text(0.5, 0.95, f"Total peaks: {len(peaks)}", transform=ax1.transAxes,
                verticalalignment='top', horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Paired hashes histogram
    votes_per_offset = []
    for t1_list in hashes_paired.values():
        votes_per_offset.extend(t1_list)
    
    if votes_per_offset:
        ax2.hist(votes_per_offset, bins=50, color='cyan', alpha=0.7)
        ax2.set_title("Paired Hashes Approach: Offset Clustering (Decisive)")
        ax2.set_xlabel("Offset Time Frame")
        ax2.set_ylabel("Vote Count")
        ax2.text(0.5, 0.95, f"Total hash pairs: {len(hashes_paired)}", transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig("report_figures/exp4_peaks_vs_hashes.png")
    plt.close()

    # 5. Pitch Shift Robustness & Sweep
    print("Generating Experiment 5: Pitch Shift Robustness & Sweep...")
    y_pitched_up = librosa.effects.pitch_shift(y_snippet, sr=sr, n_steps=3)  # +3 semitones
    y_pitched_down = librosa.effects.pitch_shift(y_snippet, sr=sr, n_steps=-3)  # -3 semitones
    
    stft_pitched_up = fp.compute_spectrogram(y_pitched_up, sr)
    stft_pitched_down = fp.compute_spectrogram(y_pitched_down, sr)
    peaks_pitched_up = fp.get_constellation_map(stft_pitched_up)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1.imshow(stft_snippet, origin='lower', aspect='auto', cmap='magma')
    ax1.set_title("Original Spectrogram (15s)")
    
    ax2.imshow(stft_pitched_up, origin='lower', aspect='auto', cmap='magma')
    ax2.set_title("Pitched Up (+3 semitones) - Frequencies Shift Higher")
    
    ax3.imshow(stft_pitched_down, origin='lower', aspect='auto', cmap='magma')
    ax3.set_title("Pitched Down (-3 semitones) - Frequencies Shift Lower")
    
    # Overlay peaks to show shift
    if peaks_clean and peaks_pitched_up:
        f_orig = [p[1] for p in peaks_clean[:300]]
        f_pitched = [p[1] for p in peaks_pitched_up[:300]]
        ax4.scatter(f_orig, [1]*len(f_orig), label='Original', alpha=0.5, s=15, color='blue')
        ax4.scatter(f_pitched, [2]*len(f_pitched), label='Pitched Up', alpha=0.5, s=15, color='orange')
        ax4.set_title("Frequency Peak Shifts: Pitch-shifted audio has different bin indices")
        ax4.legend()
        ax4.set_ylabel("Signal Version")
        ax4.set_xlabel("Frequency Bin Index")
    
    plt.tight_layout()
    plt.savefig("report_figures/exp5_pitch_shift.png")
    plt.close()

    # Pitch shift sweep
    pitch_shifts = np.array([-3.0, -2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0, 3.0])
    pitch_votes = []
    print("Running pitch shift sweep...")
    for shift in pitch_shifts:
        shifted_sig = librosa.effects.pitch_shift(y_test_clip, sr=sr, n_steps=shift)
        pred, votes = match_signal(shifted_sig, sr, db)
        is_correct = (pred == correct_song_name)
        pitch_votes.append(votes if is_correct else 0)
        print(f"  Pitch shift={shift:+.1f} semitones: Predicted='{pred}', Votes={votes} (Correct: {is_correct})")
        
    plt.figure(figsize=(8, 4.5))
    plt.plot(pitch_shifts, pitch_votes, marker='o', linewidth=2, color='crimson', label='Correct Match Votes')
    plt.title("Pitch Robustness Sweep: Matching Votes vs. Pitch Shift")
    plt.xlabel("Pitch Shift (Semitones)")
    plt.ylabel("Clustering Spike Votes (Hits Score)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("report_figures/exp5_pitch_sweep.png")
    plt.close()

    # 6. Time Stretch Robustness & Sweep
    print("Generating Experiment 6: Time Stretch Robustness & Sweep...")
    y_stretched = librosa.effects.time_stretch(y_snippet, rate=1.2)  # 20% slower
    y_squeezed = librosa.effects.time_stretch(y_snippet, rate=0.8)   # 20% faster
    
    stft_stretched = fp.compute_spectrogram(y_stretched, sr)
    stft_squeezed = fp.compute_spectrogram(y_squeezed, sr)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].imshow(stft_snippet, origin='lower', aspect='auto', cmap='viridis')
    axes[0].set_title("Original Duration")
    
    axes[1].imshow(stft_stretched, origin='lower', aspect='auto', cmap='viridis')
    axes[1].set_title("Stretched (+20% slower)")
    
    axes[2].imshow(stft_squeezed, origin='lower', aspect='auto', cmap='viridis')
    axes[2].set_title("Squeezed (-20% faster)")
    
    for ax in axes:
        ax.set_ylabel("Frequency Bin")
        ax.set_xlabel("Time Frame")
    
    plt.tight_layout()
    plt.savefig("report_figures/exp6_time_stretch.png")
    plt.close()

    # Time stretch sweep
    stretch_rates = np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
    stretch_votes = []
    print("Running time stretch sweep...")
    for rate in stretch_rates:
        stretched_sig = librosa.effects.time_stretch(y_test_clip, rate=rate)
        pred, votes = match_signal(stretched_sig, sr, db)
        is_correct = (pred == correct_song_name)
        stretch_votes.append(votes if is_correct else 0)
        print(f"  Stretch rate={rate:.2f}x: Predicted='{pred}', Votes={votes} (Correct: {is_correct})")
        
    plt.figure(figsize=(8, 4.5))
    plt.plot(stretch_rates, stretch_votes, marker='o', linewidth=2, color='goldenrod', label='Correct Match Votes')
    plt.title("Time-Stretch Robustness Sweep: Matching Votes vs. Playback Rate")
    plt.xlabel("Playback Rate (Stretch Factor)")
    plt.ylabel("Clustering Spike Votes (Hits Score)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("report_figures/exp6_time_sweep.png")
    plt.close()

    print("\nAll experiment graphics generated inside the 'report_figures/' directory successfully!")

if __name__ == "__main__":
    run_analytical_experiments()