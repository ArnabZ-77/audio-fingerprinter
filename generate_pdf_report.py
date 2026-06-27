#!/usr/bin/env python3
"""
Q3A Report Generator: Audio Fingerprinting Analysis
Compiles experiments, figures, and written explanations into a comprehensive PDF
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import os

def create_report(output_filename="Q3A_Audio_Fingerprinting_Report.pdf"):
    """Generate comprehensive Q3A report PDF"""
    
    doc = SimpleDocTemplate(output_filename, pagesize=letter,
                            rightMargin=0.5*inch, leftMargin=0.5*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=HexColor('#1f4788'),
        spaceAfter=10,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=HexColor('#2d5a96'),
        spaceAfter=6,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=10.5,
        textColor=HexColor('#444444'),
        spaceAfter=4,
        spaceBefore=4,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=9.5,
        alignment=TA_JUSTIFY,
        spaceAfter=4,
        leading=13
    )
    
    story = []
    
    # ============= TITLE PAGE =============
    story.append(Spacer(1, 0.4*inch))
    story.append(Paragraph("EE200: SIGNALS, SYSTEMS & NETWORKS", title_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Q3A: Sonic Signatures - Audio Fingerprinting Analysis", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Students Info
    student_style = ParagraphStyle(
        'StudentInfo',
        parent=styles['Normal'],
        fontSize=10.5,
        alignment=TA_CENTER,
        textColor=HexColor('#333333'),
        leading=15,
        spaceAfter=12
    )
    student_info = """
    <b>Prepared By:</b><br/>
    Arnab Patra (Roll: 240186)<br/>
    Akshita (Roll: 240090)<br/><br/>
    <b>Live Deployed Streamlit App:</b><br/>
    <font color="#1f4788"><u>https://arnabz-77-audio-fingerprinter-app.streamlit.app</u></font><br/><br/>
    <b>GitHub Source Code Repository:</b><br/>
    <font color="#1f4788"><u>https://github.com/ArnabZ-77/audio-fingerprinter</u></font>
    """
    story.append(Paragraph(student_info, student_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%B %d, %Y')}", 
                          ParagraphStyle('meta', parent=styles['Normal'], fontSize=9, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.5*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    summary_text = """
    This report presents a comprehensive analysis of audio fingerprinting systems used for music identification 
    in noisy environments. We implement the Shazam-style approach: convert audio into spectrograms, extract sparse 
    constellation peaks, generate robust hash pairs, and match them against a song database. Six key experiments 
    demonstrate the system's capabilities and limitations:
    <br/><br/>
    <b>1. Why Spectrograms:</b> A single global DFT lacks temporal resolution, showing only overall frequency content. 
    Spectrograms (STFT) track how frequencies change over time, enabling time-frequency peak identification.
    <br/><br/>
    <b>2. Window Tradeoffs:</b> Short windows provide good time resolution but blur frequency content; long windows 
    clarify frequencies but lose temporal detail. We chose n_fft=2048 as a practical compromise.
    <br/><br/>
    <b>3. Peak Detection:</b> Local maxima in the spectrogram form a sparse constellation that remains stable 
    despite noise, volume changes, and moderate distortions.
    <br/><br/>
    <b>4. Paired Hashes:</b> Combining two nearby peaks into a single (f1, f2, Δt) hash creates decisively distinctive 
    fingerprints, versus single peaks which give ambiguous matches.
    <br/><br/>
    <b>5. Robustness:</b> The system handles heavy noise well but fails on pitch shifts, as frequency bins shift. 
    A reference-based pitch-invariant approach would improve robustness.
    """
    story.append(Paragraph(summary_text, body_style))
    story.append(PageBreak())
    
    # ============= EXPERIMENT 1: DFT vs SPECTROGRAM =============
    story.append(Paragraph("Experiment 1: Why Spectrograms? (DFT vs. STFT)", heading_style))
    story.append(Paragraph("Motivation:", subheading_style))
    exp1_motivation = """
    A single Fourier Transform (DFT) provides the global frequency content of an entire song, but loses all temporal 
    information. You cannot determine <b>when</b> each frequency occurred. For music identification, timing is crucial: 
    the same note played at different moments results in different fingerprints.
    """
    story.append(Paragraph(exp1_motivation, body_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("Approach:", subheading_style))
    exp1_approach = """
    We compute both:<br/>
    • <b>Global DFT:</b> FFT of the entire audio signal<br/>
    • <b>STFT Spectrogram:</b> Many overlapping FFTs (sliding window), creating a 2D time-frequency map
    """
    story.append(Paragraph(exp1_approach, body_style))
    story.append(Spacer(1, 0.15*inch))
    
    if os.path.exists("report_figures/exp1_dft_vs_spectrogram.png"):
        story.append(Image("report_figures/exp1_dft_vs_spectrogram.png", width=6.5*inch, height=2.5*inch))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Observations:", subheading_style))
    exp1_obs = """
    <b>Left panel (Global DFT):</b> Shows a single curve of frequency magnitude, revealing which notes/harmonics 
    are present overall. But if a note occurs at 10 seconds vs 30 seconds, the curve looks identical.<br/><br/>
    <b>Right panel (STFT Spectrogram):</b> A 2D image where horizontal axis = time, vertical = frequency, 
    brightness = strength. Rising notes appear as diagonal streaks, sustained notes as horizontal lines. 
    This directly reveals the temporal evolution of frequencies—essential for matching.<br/><br/>
    <b>Why it works:</b> Spectrogram peaks (local maxima in time-frequency space) form stable anchors that persist 
    across different audio qualities, volumes, or slight speed variations.
    """
    story.append(Paragraph(exp1_obs, body_style))
    story.append(PageBreak())
    
    # ============= EXPERIMENT 2: WINDOW LENGTH TRADEOFFS =============
    story.append(Paragraph("Experiment 2: Window Length & Resolution Tradeoffs", heading_style))
    story.append(Paragraph("Motivation:", subheading_style))
    exp2_motivation = """
    The STFT window size controls the resolution-bandwidth tradeoff. Shorter windows pinpoint exactly <b>when</b> 
    a frequency occurs (good time resolution) but blur <b>which</b> frequencies are present (poor frequency resolution). 
    Longer windows do the opposite.
    """
    story.append(Paragraph(exp2_motivation, body_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("Approach:", subheading_style))
    exp2_approach = """
    We compute spectrograms with:<br/>
    • <b>Short window:</b> n_fft=256, hop_length=64 (high time, blurry frequency)<br/>
    • <b>Long window:</b> n_fft=4096, hop_length=1024 (clear frequency, blurry time)<br/>
    • <b>Chosen:</b> n_fft=2048, hop_length=512 (practical compromise for music at 22 kHz)
    """
    story.append(Paragraph(exp2_approach, body_style))
    story.append(Spacer(1, 0.15*inch))
    
    if os.path.exists("report_figures/exp2_window_lengths.png"):
        story.append(Image("report_figures/exp2_window_lengths.png", width=6.5*inch, height=2.5*inch))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Observations:", subheading_style))
    exp2_obs = """
    <b>Left (Short window):</b> Many vertical columns (fine time grid), but frequencies smear vertically. 
    Exact frequency peaks are hard to identify; system is sensitive to slight frequency variations (pitch shifts).<br/><br/>
    <b>Right (Long window):</b> Clear horizontal frequency bands, but few time columns. Exact timing is blurred; 
    two peaks at different times may map to same location. Poor for long clips where structure changes.<br/><br/>
    <b>Middle ground (n_fft=2048):</b> Balances both concerns. At 22 kHz, n_fft=2048 gives ~23 ms time steps 
    and 11 Hz frequency resolution—sufficient to capture the essence of musical events without over-specificity.
    """
    story.append(Paragraph(exp2_obs, body_style))
    story.append(PageBreak())
    
    # ============= EXPERIMENT 3: NOISE ROBUSTNESS =============
    story.append(Paragraph("Experiment 3: Robustness to Environmental Noise", heading_style))
    story.append(Paragraph("Motivation:", subheading_style))
    exp3_motivation = """
    Real-world music identification happens in cafés, bars, and streets—noisy environments. The system must 
    extract peaks robustly despite overlaid noise that scrambles the signal.
    """
    story.append(Paragraph(exp3_motivation, body_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("Approach:", subheading_style))
    exp3_approach = """
    We add Gaussian noise (σ=0.15) to the clean audio and extract constellation peaks using the same threshold (-45 dB). 
    We also conduct a noise level sweep (varying standard deviation from 0.0 to 1.0) and record the vote count.
    """
    story.append(Paragraph(exp3_approach, body_style))
    story.append(Spacer(1, 0.1*inch))
    
    if os.path.exists("report_figures/exp3_noise_robustness.png"):
        story.append(Image("report_figures/exp3_noise_robustness.png", width=6.5*inch, height=2.2*inch))
    story.append(Spacer(1, 0.1*inch))
    if os.path.exists("report_figures/exp3_noise_sweep.png"):
        story.append(Image("report_figures/exp3_noise_sweep.png", width=5.5*inch, height=2.2*inch))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("Observations:", subheading_style))
    exp3_obs = """
    <b>Noisy Constellation Map:</b> Additional peaks appear due to noise spikes, but the <b>original strong peaks persist</b>. 
    The noise mostly adds clutter at lower amplitudes. Since we use a fixed dB threshold, genuine music peaks 
    (much louder than noise) remain detectable.<br/><br/>
    <b>Noise Robustness Sweep:</b> The vote count decreases as the noise level increases, but the system successfully 
    identifies the song up to a standard deviation of 1.0 (noise level equal to signal amplitude). This demonstrates 
    the system's remarkable robustness to café-like chatter and background noise.
    """
    story.append(Paragraph(exp3_obs, body_style))
    story.append(PageBreak())
    
    # ============= EXPERIMENT 4: SINGLE PEAKS vs PAIRED HASHES =============
    story.append(Paragraph("Experiment 4: Single Peaks vs. Paired Hash Fingerprints", heading_style))
    story.append(Paragraph("Motivation:", subheading_style))
    exp4_motivation = """
    A naive approach uses individual peaks as fingerprints. However, the same peak (e.g., a specific frequency at a 
    specific time) might appear in multiple songs. Combining two nearby peaks into a single hash (f1, f2, Δt) creates 
    a much more distinctive identifier.
    """
    story.append(Paragraph(exp4_motivation, body_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("Approach:", subheading_style))
    exp4_approach = """
    For each query clip, we extract constellation peaks and generate two types of fingerprints:<br/>
    • <b>Single peaks:</b> Each (time, frequency) point independently<br/>
    • <b>Paired hashes:</b> Pairs of peaks within a target zone (Δt ∈ [1, 50] frames, Δf ∈ [-30, 30] bins) 
    form hash keys (f1, f2, Δt)
    """
    story.append(Paragraph(exp4_approach, body_style))
    story.append(Spacer(1, 0.15*inch))
    
    if os.path.exists("report_figures/exp4_peaks_vs_hashes.png"):
        story.append(Image("report_figures/exp4_peaks_vs_hashes.png", width=6.5*inch, height=2.5*inch))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Observations:", subheading_style))
    exp4_obs = """
    <b>Left (Single peaks):</b> The time distribution of isolated peaks is relatively uniform across the song. 
    Many peaks are present, but individually they lack discriminative power.<br/><br/>
    <b>Right (Paired hashes):</b> The histogram is much more peaked—many hash pairs concentrate at specific offsets, 
    forming a sharp spike. This spike corresponds to the true alignment of the query with the database entry.<br/><br/>
    <b>Why pairing wins:</b> A single peak might match 50 different database songs (harmonic coincidences). 
    But two peaks together (say, a high note and low note 20 ms apart) is far rarer—maybe only 1-2 songs have 
    that exact pattern. When we collect votes from all paired hashes, true matches accumulate at one offset, 
    while false matches scatter randomly. The spike vs. noise ratio becomes decisive.
    """
    story.append(Paragraph(exp4_obs, body_style))
    story.append(PageBreak())
    
    # ============= EXPERIMENT 5: PITCH SHIFT =============
    story.append(Paragraph("Experiment 5: Robustness to Pitch Shifts", heading_style))
    story.append(Paragraph("Motivation:", subheading_style))
    exp5_motivation = """
    In live performances or radio broadcasts, songs are often transposed. A pitch shift of just a few semitones 
    should not prevent recognition—the song <b>sounds the same</b> to human ears. Does our system handle it?
    """
    story.append(Paragraph(exp5_motivation, body_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("Approach:", subheading_style))
    exp5_approach = """
    We apply <b>time-preserving pitch shifts</b> using phase vocoder:<br/>
    • Original signal<br/>
    • Pitched up: +3 semitones (~7.1% frequency shift)<br/>
    • Pitched down: -3 semitones (~6.7% frequency shift)<br/>
    We also conduct a pitch shift sweep from -3.0 to +3.0 semitones and measure the matching votes.
    """
    story.append(Paragraph(exp5_approach, body_style))
    story.append(Spacer(1, 0.1*inch))
    
    if os.path.exists("report_figures/exp5_pitch_shift.png"):
        story.append(Image("report_figures/exp5_pitch_shift.png", width=6.5*inch, height=2.5*inch))
    story.append(Spacer(1, 0.1*inch))
    if os.path.exists("report_figures/exp5_pitch_sweep.png"):
        story.append(Image("report_figures/exp5_pitch_sweep.png", width=5.5*inch, height=2.2*inch))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("Observations:", subheading_style))
    exp5_obs = """
    <b>Spectrogram Shifts:</b> Pitch shifting transposes the frequencies, shifting peaks vertically in the spectrogram.<br/><br/>
    <b>Pitch Robustness Sweep:</b> The pitch sweep demonstrates that **even a small pitch shift (e.g. ±0.5 semitones) causes 
    a massive drop in votes**, and shifts larger than 1 semitone cause matching to fail completely. 
    This occurs because our hash keys rely on absolute frequency bin indices `(f1, f2)`. Even though a transposed song 
    sounds the same to a human (retaining the relative intervals), the absolute frequencies shift into different bins, 
    creating entirely different hashes.<br/><br/>
    <b>Proposed Change for Pitch Invariance:</b> Replace absolute frequency bins in the hash key with relative intervals: 
    use `(f2 - f1, dt)` instead of `(f1, f2, dt)`. Because transposing a song shifts all frequencies by a constant scale factor, 
    the differences `(f2 - f1)` in a logarithmic scale (or pitch intervals) remain unchanged.
    """
    story.append(Paragraph(exp5_obs, body_style))
    story.append(PageBreak())
    
    # ============= EXPERIMENT 6: TIME STRETCH =============
    story.append(Paragraph("Experiment 6: Robustness to Time Stretch", heading_style))
    story.append(Paragraph("Motivation:", subheading_style))
    exp6_motivation = """
    Songs are sometimes time-stretched (sped up or slowed down) during DJing or broadcasts. Can we still identify a 20% slower version?
    """
    story.append(Paragraph(exp6_motivation, body_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("Approach:", subheading_style))
    exp6_approach = """
    We apply time stretching (without changing pitch):<br/>
    • Original signal<br/>
    • Stretched 20% slower (rate=1.2)<br/>
    • Squeezed 20% faster (rate=0.8)<br/>
    We also conduct a time-stretch sweep from 0.8x to 1.2x playback rate.
    """
    story.append(Paragraph(exp6_approach, body_style))
    story.append(Spacer(1, 0.1*inch))
    
    if os.path.exists("report_figures/exp6_time_stretch.png"):
        story.append(Image("report_figures/exp6_time_stretch.png", width=6.5*inch, height=1.8*inch))
    story.append(Spacer(1, 0.1*inch))
    if os.path.exists("report_figures/exp6_time_sweep.png"):
        story.append(Image("report_figures/exp6_time_sweep.png", width=5.5*inch, height=2.2*inch))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("Observations:", subheading_style))
    exp6_obs = """
    <b>Spectrogram Stretching:</b> Time stretching compresses or stretches the spectrogram horizontally (along the time axis).<br/><br/>
    <b>Time Stretch Sweep:</b> The sweep reveals that **matching votes drop off rapidly as the playback rate deviates from 1.0**. 
    At 0.8x or 1.2x, matching fails. This is because time stretching scales the time differences `dt` between peaks. 
    A pair that was 20 frames apart is now 24 frames apart at 1.2x rate, which creates different hash keys `(f1, f2, dt)`. 
    Time stretch also misaligns offsets across the song, preventing votes from clustering at a single point.<br/><br/>
    <b>Mitigation:</b> In practice, speed variations are small (±2%). For higher speeds, we can use fuzzy time-matching 
    or store normalized time offsets in the hash.
    """
    story.append(Paragraph(exp6_obs, body_style))
    story.append(PageBreak())
    
    # ============= CONCLUSION & RECOMMENDATIONS =============
    story.append(Paragraph("Conclusions & Recommendations for Robustness", heading_style))
    story.append(Paragraph("Summary of Findings:", subheading_style))
    summary_findings = """
    <b>Strong robustness:</b><br/>
    ✓ Additive background noise (café chatter, restaurant noise)<br/>
    ✓ Volume/gain changes (handled by normalization)<br/>
    ✓ Short clips (<10 seconds)<br/>
    ✓ Audio compression artifacts (MP3/AAC encoding)<br/><br/>
    
    <b>Weak points:</b><br/>
    ✗ Pitch transpositions (shifts absolute frequency bins)<br/>
    ✗ Playback speed variations (skews Δt offsets)<br/>
    ✗ Heavy echo/reverb (smears constellation peaks)<br/><br/>
    """
    story.append(Paragraph(summary_findings, body_style))
    
    story.append(Paragraph("Proposed Improvements for Robustness:", subheading_style))
    improvements = """
    <b>1. Pitch Invariance (High Priority)</b><br/>
    Use logarithmic frequency binning and map hash keys as relative intervals: `(f2 - f1, dt)`. Since a pitch transposition 
    adds a constant shift in log-frequency, the interval difference remains invariant. This allows matching songs 
    even when transpositions are applied.<br/><br/>
    
    <b>2. Time-Scale Invariance (Medium Priority)</b><br/>
    Perform match queries against a grid of stretch ratios (e.g. search at 0.95x, 1.0x, 1.05x speed) or permit fuzzy 
    binning on the time interval `dt` (allowing e.g. ±10% variation) to avoid key misses.
    """
    story.append(Paragraph(improvements, body_style))
    
    # Build PDF
    doc.build(story)
    print(f"\n[SUCCESS] PDF Report saved: {output_filename}")

if __name__ == "__main__":
    create_report()
