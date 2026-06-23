import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import fingerprint as fp

def run_analytical_experiments():
    os.makedirs("report_figures", exist_ok=True)
    songs = [f for f in os.listdir("song_library") if f.endswith(('.mp3', '.wav'))]
    
    if not songs:
        print("Please place audio tracks in 'song_library' folder before compiling metrics.")
        return
        
    target_track = os.path.join("song_library", songs[0])
    print(f"Running experiments using baseline reference audio track: {target_track}")
    y, sr = fp.load_audio(target_track)
    
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

    # 3. Noise Contamination Analysis
    print("Generating Experiment 3: Environmental Noise Robustness Simulation...")
    noise = np.random.normal(0, 0.15, len(y))
    noisy_y = y + noise
    stft_noisy = fp.compute_spectrogram(noisy_y, sr)
    peaks_clean = fp.get_constellation_map(stft_db)
    peaks_noisy = fp.get_constellation_map(stft_noisy)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.imshow(stft_db, origin='lower', aspect='auto', cmap='gray', alpha=0.3)
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
    hash_counts = {h: len(v) for h, v in hashes_paired.items()}
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

    # 5. Pitch Shift Robustness Test
    print("Generating Experiment 5: Pitch Shift Robustness...")
    y_pitched_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=3)  # +3 semitones
    y_pitched_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-3)  # -3 semitones
    
    stft_pitched_up = fp.compute_spectrogram(y_pitched_up, sr)
    stft_pitched_down = fp.compute_spectrogram(y_pitched_down, sr)
    peaks_pitched_up = fp.get_constellation_map(stft_pitched_up)
    peaks_pitched_down = fp.get_constellation_map(stft_pitched_down)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1.imshow(stft_db, origin='lower', aspect='auto', cmap='magma')
    ax1.set_title("Original Spectrogram")
    
    ax2.imshow(stft_pitched_up, origin='lower', aspect='auto', cmap='magma')
    ax2.set_title("Pitched Up (+3 semitones) - Frequencies Shift Higher")
    
    ax3.imshow(stft_pitched_down, origin='lower', aspect='auto', cmap='magma')
    ax3.set_title("Pitched Down (-3 semitones) - Frequencies Shift Lower")
    
    # Overlay peaks to show shift
    if peaks and peaks_pitched_up:
        t_orig, f_orig = zip(*peaks[:500])
        t_pitched, f_pitched = zip(*peaks_pitched_up[:500])
        ax4.scatter(f_orig, [1]*len(f_orig), label='Original', alpha=0.5, s=10)
        ax4.scatter(f_pitched, [2]*len(f_pitched), label='Pitched Up', alpha=0.5, s=10)
        ax4.set_title("Frequency Peak Shifts: Pitch-shifted audio has different bin indices")
        ax4.legend()
        ax4.set_ylabel("Signal Version")
    
    plt.tight_layout()
    plt.savefig("report_figures/exp5_pitch_shift.png")
    plt.close()

    # 6. Time Stretch Robustness Test
    print("Generating Experiment 6: Time Stretch Robustness...")
    y_stretched = librosa.effects.time_stretch(y, rate=1.2)  # 20% slower
    y_squeezed = librosa.effects.time_stretch(y, rate=0.8)   # 20% faster
    
    stft_stretched = fp.compute_spectrogram(y_stretched, sr)
    stft_squeezed = fp.compute_spectrogram(y_squeezed, sr)
    peaks_stretched = fp.get_constellation_map(stft_stretched)
    peaks_squeezed = fp.get_constellation_map(stft_squeezed)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].imshow(stft_db, origin='lower', aspect='auto', cmap='viridis')
    axes[0].set_title(f"Original Duration")
    
    axes[1].imshow(stft_stretched, origin='lower', aspect='auto', cmap='viridis')
    axes[1].set_title(f"Stretched (+20% slower)")
    
    axes[2].imshow(stft_squeezed, origin='lower', aspect='auto', cmap='viridis')
    axes[2].set_title(f"Squeezed (-20% faster)")
    
    for ax in axes:
        ax.set_ylabel("Frequency Bin")
        ax.set_xlabel("Time Frame")
    
    plt.tight_layout()
    plt.savefig("report_figures/exp6_time_stretch.png")
    plt.close()

    print("\nAll experiment graphics generated inside the 'report_figures/' directory successfully!")

if __name__ == "__main__":
    run_analytical_experiments()