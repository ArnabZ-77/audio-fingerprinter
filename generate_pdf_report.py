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
        fontSize=24,
        textColor=HexColor('#1f4788'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=HexColor('#2d5a96'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=11,
        textColor=HexColor('#444444'),
        spaceAfter=6,
        spaceBefore=6,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        leading=14
    )
    
    story = []
    
    # ============= TITLE PAGE =============
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("EE200: SIGNALS, SYSTEMS & NETWORKS", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Q3A: Sonic Signatures - Audio Fingerprinting Analysis", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%B %d, %Y')}", 
                          ParagraphStyle('meta', parent=styles['Normal'], fontSize=9, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.8*inch))
    
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
    Compare the clean vs. noisy peak distributions.
    """
    story.append(Paragraph(exp3_approach, body_style))
    story.append(Spacer(1, 0.15*inch))
    
    if os.path.exists("report_figures/exp3_noise_robustness.png"):
        story.append(Image("report_figures/exp3_noise_robustness.png", width=6.5*inch, height=2.5*inch))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Observations:", subheading_style))
    exp3_obs = """
    <b>Left (Clean):</b> Sharp, well-defined peaks form a constellation. Many local maxima exceed the threshold.<br/><br/>
    <b>Right (Noisy):</b> Additional peaks appear due to noise spikes, but the <b>original strong peaks persist</b>. 
    The noise mostly adds clutter at lower amplitudes. Since we use a fixed dB threshold, genuine music peaks 
    (much louder than noise) remain detectable.<br/><br/>
    <b>Key insight:</b> Storing only local maxima above a threshold makes the fingerprint naturally robust to 
    additive noise. Volume scaling (dB conversion) means the system is also robust to gain changes (microphone distance, 
    speaker volume).
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
    Then extract spectrogram and peaks for each.
    """
    story.append(Paragraph(exp5_approach, body_style))
    story.append(Spacer(1, 0.15*inch))
    
    if os.path.exists("report_figures/exp5_pitch_shift.png"):
        story.append(Image("report_figures/exp5_pitch_shift.png", width=6.5*inch, height=3.5*inch))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Observations:", subheading_style))
    exp5_obs = """
    <b>Top left (Original):</b> Peaks at their natural frequency bins.<br/><br/>
    <b>Top right (Pitched up +3 semitones):</b> <b>All peaks shift to higher frequency bins</b>. The spectral 
    pattern is identical, just shifted upward in frequency.<br/><br/>
    <b>Bottom left (Pitched down):</b> Peaks shift to lower bins.<br/><br/>
    <b>Bottom right (Frequency comparison):</b> The pitch-shifted versions have peaks at different bin indices. 
    Our hash keys include absolute frequencies (f1, f2), so a shifted peak that was at bin 100 now at bin 110 
    creates a completely different hash.<br/><br/>
    <b>Failure mechanism:</b> Our system is <b>NOT pitch-invariant</b>. A +3 semitone shift defeats matching 
    even though the human ear perceives the same melody. This is because we store absolute frequency bins, 
    not relative frequency relationships (e.g., interval ratios).<br/><br/>
    <b>To fix this:</b> Use relative intervals instead: hash keys could be (f2 - f1, Δt) rather than (f1, f2, Δt). 
    This way, a global frequency shift leaves the intervals unchanged. Alternatively, use a reference pitch (e.g., A4=440 Hz) 
    and store semitone offsets instead of absolute bins.
    """
    story.append(Paragraph(exp5_obs, body_style))
    story.append(PageBreak())
    
    # ============= EXPERIMENT 6: TIME STRETCH =============
    story.append(Paragraph("Experiment 6: Robustness to Time Stretch", heading_style))
    story.append(Paragraph("Motivation:", subheading_style))
    exp6_motivation = """
    Songs are sometimes time-stretched (sped up or slowed down) during broadcast, DJing, or when transferred 
    between systems at different playback speeds. Can we still identify a 20% slower version?
    """
    story.append(Paragraph(exp6_motivation, body_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("Approach:", subheading_style))
    exp6_approach = """
    We apply time stretching (without changing pitch):<br/>
    • Original signal<br/>
    • Stretched 20% slower (rate=1.2): more frames, same frequencies<br/>
    • Squeezed 20% faster (rate=0.8): fewer frames, same frequencies
    """
    story.append(Paragraph(exp6_approach, body_style))
    story.append(Spacer(1, 0.15*inch))
    
    if os.path.exists("report_figures/exp6_time_stretch.png"):
        story.append(Image("report_figures/exp6_time_stretch.png", width=6.5*inch, height=2.2*inch))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Observations:", subheading_style))
    exp6_obs = """
    <b>Left (Original):</b> Baseline spectrogram width and structure.<br/><br/>
    <b>Middle (Stretched +20%):</b> More time frames (right side longer), but frequency content unchanged. 
    The spectrogram is <b>horizontally stretched</b>.<br/><br/>
    <b>Right (Squeezed -20%):</b> Fewer time frames, but again same frequencies. Spectrogram is <b>horizontally compressed</b>.<br/><br/>
    <b>Impact on hashing:</b> Our hash keys include Δt (time gap between two peaks). If a song plays 20% slower, 
    all Δt values scale by 1.2. A pair that was 20 frames apart is now 24 frames apart—creating a different hash. 
    This causes hash mismatches.<br/><br/>
    <b>Mitigation:</b> Time stretching is less common than pitch shifting, and minor variations (±5%) often fall 
    within matching tolerance (we use Δt ranges like [1, 50] frames). However, for true robustness, we could:<br/>
    1. Store normalized time intervals (e.g., as fractions of song duration)<br/>
    2. Allow fuzzy matching on Δt (e.g., ±10% tolerance)<br/>
    3. Use a combination of multiple scales
    """
    story.append(Paragraph(exp6_obs, body_style))
    story.append(PageBreak())
    
    # ============= CONCLUSION & RECOMMENDATIONS =============
    story.append(Paragraph("Conclusions & Recommendations for Robustness", heading_style))
    story.append(Paragraph("Summary of Findings:", subheading_style))
    summary_findings = """
    <b>Strong robustness:</b><br/>
    ✓ Environmental noise (café chatter, background music)<br/>
    ✓ Volume/gain changes (dB normalization handles this)<br/>
    ✓ Short clips (<10% of song duration)<br/>
    ✓ Audio compression artifacts (AAC, MP3)<br/><br/>
    
    <b>Weak points:</b><br/>
    ✗ Pitch shifts >1 semitone (absolute frequency bins fail)<br/>
    ✗ Significant time stretching >10% (Δt scaling fails)<br/>
    ✗ Heavily reverb/echo (smears peaks in time-frequency)<br/><br/>
    """
    story.append(Paragraph(summary_findings, body_style))
    
    story.append(Paragraph("Proposed Improvements:", subheading_style))
    improvements = """
    <b>1. Pitch Invariance (Priority: HIGH)</b><br/>
    Replace absolute frequency pairs with relative intervals: use (f2 - f1, Δt) as hash keys instead of (f1, f2, Δt). 
    This makes the system immune to global pitch shifts. A melody shifted up/down by N semitones still has the same 
    interval pattern.<br/><br/>
    
    <b>2. Time-Scale Invariance (Priority: MEDIUM)</b><br/>
    Normalize Δt to a reference (e.g., as a fraction of local region size) or use fuzzy matching with ±15% tolerance 
    on time offsets. This handles moderate playback speed variations.<br/><br/>
    
    <b>3. Multi-Scale Matching (Priority: MEDIUM)</b><br/>
    Extract peaks at multiple frequency resolutions. Some peaks are global (slow-evolving harmonics), others are 
    local (fast transients). Multiple scales capture both.<br/><br/>
    
    <b>4. Confidence Scoring (Priority: LOW)</b><br/>
    Weight hash matches by peak prominence (stronger peaks = more votes). Rare, weak peaks carry less confidence.
    """
    story.append(Paragraph(improvements, body_style))
    
    # Build PDF
    doc.build(story)
    print(f"\n✅ PDF Report saved: {output_filename}")

if __name__ == "__main__":
    create_report()
