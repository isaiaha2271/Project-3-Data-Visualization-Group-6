# Visual Attention Patterns in Successful vs Unsuccessful ILS Approaches

**CECS 450 – Fall 2025 – Project 3 (Option A)**

## Overview

This interactive dashboard visualizes eye-tracking data from 39 pilots performing ILS (Instrument Landing System) approaches in a flight simulator. The dashboard compares visual attention patterns between successful pilots (Approach_Score ≥ 0.7) and unsuccessful pilots (< 0.7).

## Key Findings

**Successful pilots demonstrate:**
- More attention to the Attitude Indicator (AI) and Horizontal Situation Indicator (HSI)
- More structured scan patterns
- Lower entropy (more focused attention)
- Less time looking outside or at non-critical instruments

## Features

The dashboard includes the following visualizations:

3. **Transition Matrix Heatmaps**: AOI-to-AOI transition probabilities for both groups (using real pattern data from Excel)
4. **Most Frequent Scanpath Patterns**: Top 5-8 most common AOI sequences with frequency comparison
5. **Saccade & Fixation Summary Metrics**: Boxplots comparing key metrics (fixation duration, saccade amplitude, entropy, etc.)

## AOI Mapping

- **A**: No AOI (outside defined regions)
- **B**: Alt_VSI (Altitude/Vertical Speed Indicator)
- **C**: AI (Attitude Indicator)
- **D**: TI_HSI (Turn Indicator/Horizontal Situation Indicator)
- **E**: SSI (Slip/Skid Indicator)
- **F**: ASI (Airspeed Indicator)
- **G**: RPM (Engine RPM)
- **H**: Window (Outside view)