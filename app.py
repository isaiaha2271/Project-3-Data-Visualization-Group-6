"""
This dashboard visualizes eye-tracking data from pilots performing ILS approaches,
comparing successful (Approach_Score ≥ 0.7) vs unsuccessful pilots.

Design Principles Applied:
- Colorblind-safe palettes (green for successful, red for unsuccessful)
- Preattentive attributes for highlighting differences
- Gestalt principles for grouping
- Reduced chartjunk
- Support for comparison tasks
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
import os

app = dash.Dash(__name__)
app.title = "ILS Approach Visual Attention Dashboard"

# Load data
DATA_PATH = os.path.join('Data', 'AOI_DGMs.csv')

# Pattern data from Excel files (with multiple sheets)
PATTERNS_EXCEL_PATH = os.path.join('Data', 'Collapsed Patterns (Group).xlsx')
EXPANDED_PATTERNS_EXCEL_PATH = os.path.join('Data', 'Expanded Patterns (Group).xlsx')

# Load main data
df = pd.read_csv(DATA_PATH)

# Load pattern data from Excel files
# Excel file structure:
# - Collapsed Patterns (Group).xlsx:
#   * Sheet "Succesful" - Successful pilots' collapsed patterns
#   * Sheet "Unsuccesful" - Unsuccessful pilots' collapsed patterns
#   * Sheet "Succesful Excluding No AOI(A)" - Successful patterns without No AOI
#   * Sheet "Unsuccesful Excluding No AOI(A)" - Unsuccessful patterns without No AOI
# - Expanded Patterns (Group).xlsx: Same sheet structure

# Load successful patterns from Excel (sheet: "Succesful")
patterns_success = pd.read_excel(PATTERNS_EXCEL_PATH, sheet_name='Succesful')

# Load unsuccessful patterns from Excel (sheet: "Unsuccesful")
try:
    patterns_unsuccess = pd.read_excel(PATTERNS_EXCEL_PATH, sheet_name='Unsuccesful')
except Exception as e:
    patterns_unsuccess = None
    print(f"Warning: Could not load unsuccessful patterns from Excel: {e}")

# AOI mapping: A=NoAOI, B=Alt_VSI, C=AI, D=TI_HSI, E=SSI, F=ASI, G=RPM, H=Window
AOI_MAPPING = {
    'A': 'No AOI',
    'B': 'Alt_VSI',
    'C': 'AI',
    'D': 'TI_HSI',
    'E': 'SSI',
    'F': 'ASI',
    'G': 'RPM',
    'H': 'Window'
}

AOI_ORDER = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
AOI_LABELS = [AOI_MAPPING[aoi] for aoi in AOI_ORDER]

# Color scheme 
COLORS = {
    'successful': '#2ca02c',  # Green
    'unsuccessful': '#d62728',  # Red
    'shared': '#9467bd',  # Purple
    'background': '#f7f7f7'
}

# Columns to include in the parallel coordinates plot.
PCP_COLUMNS = {
    # Fixation counts for each AOI
    'Total Fixation Count': "Total_Number_of_Fixations",
    'AI Fixation Count': 'AI_Total_Number_of_Fixations',
    'HSI Fixation Count': 'TI_HSI_Total_Number_of_Fixations',
    'SSI Fixation Count': 'SSI_Total_Number_of_Fixations',
    'ASI Fixation Count': 'ASI_Total_Number_of_Fixations',
    'RPM Fixation Count': 'RPM_Total_Number_of_Fixations',
    'Window Fixation Count': 'Window_Total_Number_of_Fixations',
    'Alt/VSI Fixation Count': 'Alt_VSI_Total_Number_of_Fixations',
    'No AOI Fixation Count': 'NoAOI_Total_Number_of_Fixations',

    # Median fixations for each AOI
    'Median Fixation Duration (s)': "Median_fixation_duration_s",
    'Median AI Fixations': 'AI_Median_fixation_duration_s',
    'Median HSI Fixations': 'TI_HSI_Median_fixation_duration_s',
    'Median SSI Fixations': 'SSI_Median_fixation_duration_s',
    'Median ASI Fixations': 'ASI_Median_fixation_duration_s',
    'Median RPM Fixations': 'RPM_Median_fixation_duration_s',
    'Median Window Fixations': 'Window_Median_fixation_duration_s',
    'Median Alt/VSI Fixations': 'Alt_VSI_Median_fixation_duration_s',
    'Median No AOI Fixations': 'NoAOI_Median_fixation_duration_s',



    # Proportion fixations for each AOI
    "Proportion of Fixations on Alt_VSI":"Alt_VSI_Proportion_of_fixations_spent_in_AOI",
    "Proportion of Fixations on NoAOI":"NoAOI_Proportion_of_fixations_spent_in_AOI",
    "Proportion of Fixations on Window":"Window_Proportion_of_fixations_spent_in_AOI",
    "Proportion of Fixations on AI":"AI_Proportion_of_fixations_spent_in_AOI",
    "Proportion of Fixations on ASI":"ASI_Proportion_of_fixations_spent_in_AOI",
    "Proportion of Fixations on SSI":"SSI_Proportion_of_fixations_spent_in_AOI",
    "Proportion of Fixations on TI_HSI":"TI_HSI_Proportion_of_fixations_spent_in_AOI",
    "Proportion of Fixations on RPM":"RPM_Proportion_of_fixations_spent_in_AOI",
    
    "Transition Entropy":"transition_entropy",
    "Approach Score":"Approach_Score",
    "Pilot Success":"pilot_success",

}


# Prepare data
df['pilot_success'] = df['pilot_success'].str.strip()
successful_df = df[df['pilot_success'] == 'Successful'].copy()
unsuccessful_df = df[df['pilot_success'] == 'Unsuccessful'].copy()

# Extract AOI proportions
def get_aoi_proportions(df_group, metric='fixations'):
    """Extract proportion data for each AOI"""
    proportions = {}
    for aoi_code, aoi_name in AOI_MAPPING.items():
        if aoi_code == 'A':
            col_name = f'NoAOI_Proportion_of_fixations_spent_in_AOI' if metric == 'fixations' else f'NoAOI_Proportion_of_fixations_durations_spent_in_AOI'
        else:
            col_name = f'{aoi_name}_Proportion_of_fixations_spent_in_AOI' if metric == 'fixations' else f'{aoi_name}_Proportion_of_fixations_durations_spent_in_AOI'
        
        if col_name in df_group.columns:
            proportions[aoi_code] = df_group[col_name].fillna(0).values
        else:
            proportions[aoi_code] = np.zeros(len(df_group))
    return proportions

# Calculate transition matrices from patterns data
def calculate_transition_matrix(patterns_df, weight_by_frequency=True):
    """Calculate AOI-to-AOI transition probabilities from patterns"""
    transitions = np.zeros((8, 8), dtype=float)  # 8 AOIs, ensure float type
    total_transitions = 0.0
    
    for idx, row in patterns_df.iterrows():
        pattern = str(row['Pattern String']).strip()
        if weight_by_frequency:
            # Ensure frequency is numeric - handle various types from Excel
            freq_val = row.get('Frequency', 1)
            try:
                # Try direct conversion first
                if pd.isna(freq_val) or freq_val == '' or freq_val is None:
                    frequency = 1.0
                else:
                    frequency = float(freq_val)
            except (ValueError, TypeError, AttributeError):
                # If conversion fails, default to 1.0
                frequency = 1.0
        else:
            frequency = 1.0
        
        if len(pattern) >= 2:
            for i in range(len(pattern) - 1):
                from_aoi = pattern[i]
                to_aoi = pattern[i + 1]
                if from_aoi in AOI_ORDER and to_aoi in AOI_ORDER:
                    from_idx = AOI_ORDER.index(from_aoi)
                    to_idx = AOI_ORDER.index(to_aoi)
                    if from_idx != to_idx:  # Exclude self-transitions
                        transitions[from_idx, to_idx] += frequency
                        total_transitions += frequency
    
    # Normalize to probabilities
    if total_transitions > 0:
        # Ensure both are proper numeric types
        total_transitions_float = float(total_transitions)
        # Convert transitions to float array explicitly
        transitions_float = transitions.astype(np.float64)
        transitions = transitions_float / np.float64(total_transitions_float)
    
    return transitions.astype(np.float64)

# Load transition matrices
transitions_success = calculate_transition_matrix(patterns_success, weight_by_frequency=True)

if patterns_unsuccess is not None and len(patterns_unsuccess) > 0:
    transitions_unsuccess = calculate_transition_matrix(patterns_unsuccess, weight_by_frequency=True)
else:
    # Fallback: derive from proportion data
    unsuccess_dur_prop = get_aoi_proportions(unsuccessful_df, 'durations')
    proportions = []
    for aoi in AOI_ORDER:
        if len(unsuccess_dur_prop[aoi]) > 0:
            mean_val = np.mean(unsuccess_dur_prop[aoi])
            # Ensure it's a scalar float
            if isinstance(mean_val, np.ndarray):
                mean_val = float(mean_val.item())
            else:
                mean_val = float(mean_val)
            proportions.append(mean_val)
        else:
            proportions.append(0.0)
    
    total_prop = float(sum(proportions))
    if total_prop > 0:
        proportions = [float(p) / total_prop for p in proportions]
        transitions_unsuccess = np.zeros((8, 8), dtype=float)
        for i in range(8):
            for j in range(8):
                if i != j:
                    transitions_unsuccess[i, j] = float(proportions[j])
        for i in range(8):
            row_sum = float(np.sum(transitions_unsuccess[i, :]))
            if row_sum > 0:
                row_array = transitions_unsuccess[i, :].astype(np.float64)
                transitions_unsuccess[i, :] = row_array / np.float64(row_sum)
    else:
        transitions_unsuccess = np.zeros((8, 8), dtype=float)

# Get top patterns for each group
def get_top_patterns(patterns_df, n=8):
    """Get top N patterns by frequency"""
    top = patterns_df.nlargest(n, 'Frequency')
    result = []
    for idx, row in top.iterrows():
        result.append({
            'Pattern String': row['Pattern String'],
            'Frequency': row['Frequency'],
            'Proportional Pattern Frequency': row.get('Proportional Pattern Frequency', 0)
        })
    return result



#normalize certain cols in df_group and return nromalized df
def normalize_pcp_df(df_group,cols):
    df_norm = df_group.copy()
    for c in cols:
        #skip any cols that dont appear in df
        if c not in df_group:
            continue
        col = df_norm[c].astype(float) #convert column values to float
        min_val = col.min()
        max_val = col.max()

        #check if max,min values are valid and if col contains all repeating values
        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            df_norm[c] = 0.0
        else:
            df_norm[c] = (col - min_val) / (max_val - min_val) #normalize the column
    return df_norm


    
#"""Sample rows if data frame is large(>300 rows)"""
def _maybe_sample(df_group, max_rows=300):
    n = len(df_group)

    if n <= max_rows:
        return df_group

    # random sample but keep reproducible ordering
    return df_group.sample(n=max_rows, random_state=42)







top_success_patterns = get_top_patterns(patterns_success, 8)
if patterns_unsuccess is not None and len(patterns_unsuccess) > 0:
    top_unsuccess_patterns = get_top_patterns(patterns_unsuccess, 8)
else:
    top_unsuccess_patterns = []

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Visual Attention Patterns in Successful vs Unsuccessful ILS Approaches", 
                style={'textAlign': 'center', 'color': '#1f77b4', 'marginBottom': '10px'}),
        html.P("CECS 450 – Fall 2025 – Project 3 (Option A)", 
               style={'textAlign': 'center', 'color': '#666', 'fontSize': '14px', 'marginBottom': '30px'}),
        html.P("Successful pilots look more at the Attitude Indicator and HSI, follow structured scan patterns, have lower entropy, and spend less time looking outside or at non-critical instruments.",
               style={'textAlign': 'center', 'color': '#333', 'fontSize': '16px', 'fontStyle': 'italic', 
                      'marginBottom': '30px', 'padding': '15px', 'backgroundColor': '#e8f4f8', 'borderRadius': '5px'})
    ], style={'padding': '20px', 'backgroundColor': 'white'}),
    
    # Global Filters
    html.Div([
        html.Div([
            html.Label("Select Pilots (Multi-select):", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='pilot-filter',
                options=[{'label': pid, 'value': pid} for pid in sorted(df['PID'].unique())],
                value=list(df['PID'].unique()),
                multi=True,
                style={'width': '100%'}
            )
        ], style={'width': '100%', 'marginBottom': '20px'})
    ], style={'padding': '20px', 'backgroundColor': COLORS['background']}),
    
    # Transition Matrix Heatmaps
    html.Div([
        html.H2("AOI-to-AOI Transition Probabilities", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div([
            dcc.Graph(id='transition-heatmaps')
        ])
    ], style={'padding': '20px', 'marginBottom': '20px'}),
    
    # Most Frequent Scanpath Patterns
    html.Div([
        html.H2("Most Frequent Scanpath Patterns", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div([
            dcc.Graph(id='scanpath-patterns')
        ])
    ], style={'padding': '20px', 'marginBottom': '20px'}),


    # Parallel Coordinate Plots
    html.Div([
        html.H2("Parallel Coordinate Comparison: Successful vs Unsuccessful", style={'textAlign': 'center', 'marginBottom': '12px'}),
        html.Div([
            html.Div(dcc.Graph(id='pcp-success'), style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div(dcc.Graph(id='pcp-unsuccess'), style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'}),
        ])
    ], style={'padding': '20px', 'marginBottom': '20px'}),

              

    
    # Saccade & Fixation Summary Metrics
    html.Div([
        html.H2("Saccade & Fixation Summary Metrics", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div([
            dcc.Graph(id='metrics-boxplots')
        ])
    ], style={'padding': '20px', 'marginBottom': '20px'})
], style={'backgroundColor': COLORS['background']})

# Callback for Transition Heatmaps
@callback(
    Output('transition-heatmaps', 'figure'),
    Input('pilot-filter', 'value')
)
def update_transition_heatmaps(selected_pilots):
    """Update transition matrix heatmaps"""
    # Use pre-calculated matrices (filtering by pilot would require recalculating)
    transitions_success_filtered = transitions_success
    transitions_unsuccess_filtered = transitions_unsuccess
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Successful Pilots', 'Unsuccessful Pilots'),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]],
        horizontal_spacing=0.2
    )
    
    # Successful heatmap
    fig.add_trace(
        go.Heatmap(
            z=transitions_success_filtered,
            x=AOI_LABELS,
            y=AOI_LABELS,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(x=0.45, len=0.4)
        ),
        row=1, col=1
    )
    
    # Unsuccessful heatmap
    fig.add_trace(
        go.Heatmap(
            z=transitions_unsuccess_filtered,
            x=AOI_LABELS,
            y=AOI_LABELS,
            colorscale='Viridis',
            showscale=False
        ),
        row=1, col=2
    )
    
    title_text = "AOI-to-AOI Transition Probabilities (Excluding Self-Transitions)"
    if patterns_unsuccess is not None and len(patterns_unsuccess) > 0:
        title_text += "<br><sub>✓ Both groups use REAL scanpath patterns from Excel files</sub>"
    
    fig.update_layout(
        height=600,
        title_text=title_text,
        title_x=0.5
    )
    
    fig.update_xaxes(title_text="To AOI", row=1, col=1)
    fig.update_xaxes(title_text="To AOI", row=1, col=2)
    fig.update_yaxes(title_text="From AOI", row=1, col=1)
    fig.update_yaxes(title_text="From AOI", row=1, col=2)
    
    return fig

# Callback for Scanpath Patterns
@callback(
    Output('scanpath-patterns', 'figure'),
    Input('pilot-filter', 'value')
)
def update_scanpath_patterns(selected_pilots):
    """Update scanpath patterns visualization"""
    # Get top patterns for both groups
    top_success = top_success_patterns[:8]
    top_unsuccess = top_unsuccess_patterns[:8] if len(top_unsuccess_patterns) > 0 else []
    
    # Create combined visualization
    all_patterns = {}
    
    # Add successful patterns
    for p in top_success:
        pattern = p['Pattern String']
        if pattern not in all_patterns:
            all_patterns[pattern] = {'success': 0, 'unsuccess': 0}
        all_patterns[pattern]['success'] = p['Frequency']
    
    # Add unsuccessful patterns
    for p in top_unsuccess:
        pattern = p['Pattern String']
        if pattern not in all_patterns:
            all_patterns[pattern] = {'success': 0, 'unsuccess': 0}
        all_patterns[pattern]['unsuccess'] = p['Frequency']
    
    # Sort by total frequency and take top 8
    sorted_patterns = sorted(all_patterns.items(), 
                           key=lambda x: x[1]['success'] + x[1]['unsuccess'], 
                           reverse=True)[:8]
    
    patterns = [p[0] for p in sorted_patterns]
    success_freqs = [all_patterns[p]['success'] for p in patterns]
    unsuccess_freqs = [all_patterns[p]['unsuccess'] for p in patterns]
    
    # Create horizontal bar chart with grouped bars
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=patterns,
        x=success_freqs,
        orientation='h',
        marker_color=COLORS['successful'],
        name='Successful',
        text=[f"{f}" if f > 0 else "" for f in success_freqs],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        y=patterns,
        x=unsuccess_freqs,
        orientation='h',
        marker_color=COLORS['unsuccessful'],
        name='Unsuccessful',
        text=[f"{f}" if f > 0 else "" for f in unsuccess_freqs],
        textposition='auto'
    ))
    
    fig.update_layout(
        barmode='group',
        height=500,
        title_text="Top 8 Most Frequent Scanpath Patterns Comparison",
        title_x=0.5,
        xaxis_title="Frequency",
        yaxis_title="Pattern Sequence",
        showlegend=True
    )
    
    return fig

# Callback for Metrics Boxplots
@callback(
    Output('metrics-boxplots', 'figure'),
    Input('pilot-filter', 'value')
)
def update_metrics_boxplots(selected_pilots):
    """Update metrics boxplots"""
    filtered_df = df[df['PID'].isin(selected_pilots)]
    successful_filtered = filtered_df[filtered_df['pilot_success'] == 'Successful']
    unsuccessful_filtered = filtered_df[filtered_df['pilot_success'] == 'Unsuccessful']
    
    # Metrics to compare
    metrics = {
        'Mean Fixation Duration (s)': 'Mean_fixation_duration_s',
        'Mean Saccade Amplitude (°)': 'mean_absolute_degree',
        'Fixation-to-Saccade Ratio': 'fixation_to_saccade_ratio',
        'Stationary Entropy': 'stationary_entropy',
        'Transition Entropy': 'transition_entropy',
        'Average Pupil Size': 'average_pupil_size_of_both_eyes'
    }
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=list(metrics.keys()),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    row = 1
    col = 1
    for metric_name, metric_col in metrics.items():
        if metric_col in filtered_df.columns:
            success_vals = successful_filtered[metric_col].dropna()
            unsuccess_vals = unsuccessful_filtered[metric_col].dropna()
            
            fig.add_trace(
                go.Box(y=success_vals, name='Successful', marker_color=COLORS['successful'], showlegend=(row==1 and col==1)),
                row=row, col=col
            )
            fig.add_trace(
                go.Box(y=unsuccess_vals, name='Unsuccessful', marker_color=COLORS['unsuccessful'], showlegend=(row==1 and col==1)),
                row=row, col=col
            )
        
        col += 1
        if col > 3:
            col = 1
            row += 1
    
    fig.update_layout(
        height=800,
        title_text="Saccade & Fixation Summary Metrics Comparison",
        title_x=0.5,
        showlegend=True
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)

