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
from dash import dcc, html, Input, Output, ctx, callback
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

# Instrument Descriptions based on Project PDF
INSTRUMENT_INFO = {
    'AI': {
        'name': 'Attitude Indicator', 
        'desc': "Displays the aircraft's orientation relative to the horizon (pitch and bank)."
    },
    'Alt_VSI': {
        'name': 'Altimeter & Vertical Speed', 
        'desc': "Provides altitude and rate-of-climb/descent information."
    },
    'ASI': {
        'name': 'Airspeed Indicator', 
        'desc': "Shows the aircraft's speed relative to surrounding air."
    },
    'SSI': {
        'name': 'Slip Skid Indicator', 
        'desc': "Indicates lateral coordination during turns."
    },
    'TI_HSI': {
        'name': 'Turn Indicator / HSI', 
        'desc': "Displays turn rate and navigational alignment (localizer tracking)."
    },
    'RPM': {
        'name': 'Tachometer', 
        'desc': "Reflects engine performance in revolutions per minute."
    },
    'Window': {
        'name': 'External View', 
        'desc': "The visual flight deck view outside the aircraft."
    },
    'No AOI': {
        'name': 'No Area of Interest', 
        'desc': "Gaze did not fall within a pre-defined instrument boundary."
    }
}

# Columns to include in the parallel coordinates plot.
PCP_COLUMNS = {
    # Fixation counts for each AOI
    #"PID":"PID",
    #'Fix. Cnt': "Total_Number_of_Fixations",
    #'AI Fix. Cnt': 'AI_Total_Number_of_Fixations',
    #'HSI Fix. Cnt': 'TI_HSI_Total_Number_of_Fixations',
    #'SSI Fix. Cnt': 'SSI_Total_Number_of_Fixations',
    #'ASI Fix. Cnt': 'ASI_Total_Number_of_Fixations',
    #'RPM Fix. Cnt': 'RPM_Total_Number_of_Fixations',
    #'Window Fix. Cnt': 'Window_Total_Number_of_Fixations',
    #'Alt/VSI Fix. Cnt': 'Alt_VSI_Total_Number_of_Fixations',
    #'No AOI Fix. Cnt': 'NoAOI_Total_Number_of_Fixations',

    #'Median fixations for each AOI
    'AI Dur. % ': 'AI_Proportion_of_fixations_durations_spent_in_AOI',
    'HSI Dur. %': 'TI_HSI_Proportion_of_fixations_durations_spent_in_AOI',
    'SSI Dur. % ': 'SSI_Proportion_of_fixations_durations_spent_in_AOI',
    'ASI Dur. % ': 'ASI_Proportion_of_fixations_durations_spent_in_AOI',
    'RPM Dur. % ': 'RPM_Proportion_of_fixations_durations_spent_in_AOI',
    'Window Dur. % ': 'Window_Proportion_of_fixations_durations_spent_in_AOI',
    'Alt/VSI Dur. % ': 'Alt_VSI_Proportion_of_fixations_durations_spent_in_AOI',
    'No AOI Dur. % ': 'NoAOI_Proportion_of_fixations_durations_spent_in_AOI',



    # Proportion fixations for each AOI
    "Alt_VSI Prop. ":"Alt_VSI_Proportion_of_fixations_spent_in_AOI",
    "NoAOI Prop. on":"NoAOI_Proportion_of_fixations_spent_in_AOI",
    "Window Prop.":"Window_Proportion_of_fixations_spent_in_AOI",
    "AI Prop.":"AI_Proportion_of_fixations_spent_in_AOI",
    "ASI Prop.":"ASI_Proportion_of_fixations_spent_in_AOI",
    "SSI Prop.":"SSI_Proportion_of_fixations_spent_in_AOI",
    "TI_HSI Prop.":"TI_HSI_Proportion_of_fixations_spent_in_AOI",
    "RPM Prop.":"RPM_Proportion_of_fixations_spent_in_AOI",
    
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
        if c not in df_group or not pd.api.types.is_numeric_dtype(df[c]):
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


    
'''#"""Sample rows if data frame is large(>300 rows)"""
def _maybe_sample(df_group, max_rows=300):
    n = len(df_group)

    if n <= max_rows:
        return df_group

    # random sample but keep reproducible ordering
    return df_group.sample(n=max_rows, random_state=42)


'''




top_success_patterns = get_top_patterns(patterns_success, 8)
if patterns_unsuccess is not None and len(patterns_unsuccess) > 0:
    top_unsuccess_patterns = get_top_patterns(patterns_unsuccess, 8)
else:
    top_unsuccess_patterns = []

def create_instrument_guide():
    return html.Details(
        [
            # 1. The Toggle Button (Summary)
            html.Summary(
                "ℹ️ Instrument Quick Guide",
                style={
                    'cursor': 'pointer',
                    'fontWeight': 'bold',
                    'fontSize': '16px',
                    'color': '#1f77b4',
                    'listStyle': 'none',       # Hides the default triangle arrow
                    'textAlign': 'left',      # Aligns text to the right
                    'padding': '5px',
                    'userSelect': 'none',      # Prevents highlighting the text when clicking
                    'backgroundColor': 'white', # Background for the button itself
                    'borderRadius': '5px',
                    'boxShadow': '0 1px 3px rgba(0,0,0,0.2)' # Subtle shadow for the button
                }
            ),
            
            # 2. The Content (Dropdown Info)
            html.Div([
                html.Div([
                    html.Span(
                        f"{abbr} ({data['name']}): ",
                        style={'fontWeight': 'bold', 'color': '#333'}
                    ),
                    html.Span(data['desc'], style={'fontSize': '13px', 'color': '#555'})
                ], style={'marginBottom': '8px', 'borderBottom': '1px solid #eee', 'paddingBottom': '4px'})
                for abbr, data in INSTRUMENT_INFO.items()
            ], style={
                'marginBottom': '10px',      # Adds space between the list and the button
                'maxHeight': '400px',        # Limits height so it doesn't cover the whole screen
                'overflowY': 'auto',         # Adds scrollbar if list is too long
                'backgroundColor': 'white',
                'padding': '15px',
                'borderRadius': '8px',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.15)',
                'border': '1px solid #ddd'
            })
        ],
        style={
            # Anchor to Bottom-Left
            'position': 'fixed',
            'bottom': '20px',
            'left': '20px',
            'width': '320px',
            'zIndex': '1000',
            'display': 'flex',
            'flexDirection': 'column-reverse', 
            'alignItems': 'stretch' 
        }
    )

# App layout
app.layout = html.Div([
    # pilot-filter
    dcc.Store(id='pilot-filter', data=list(df['PID'].unique())),
    
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
    
    # # Global Filters
    # html.Div([
    #     html.Div([
    #         html.Label("Select Pilots (Multi-select):", style={'fontWeight': 'bold', 'marginRight': '10px'}),
    #         dcc.Dropdown(
    #             id='pilot-filter',
    #             options=[{'label': pid, 'value': pid} for pid in sorted(df['PID'].unique())],
    #             value=list(df['PID'].unique()),
    #             multi=True,
    #             style={'width': '100%'}
    #         )
    #     ], style={'width': '100%', 'marginBottom': '20px'})
    # ], style={'padding': '20px', 'backgroundColor': COLORS['background']}),
    
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
        ]),
        html.Div([
            html.Div([
                html.Span("A", style={'fontWeight': 'bold', 'color': '#1f77b4', 'marginRight': '5px'}),
                html.Span("= No AOI", style={'marginRight': '20px'}),
                html.Span("B", style={'fontWeight': 'bold', 'color': '#1f77b4', 'marginRight': '5px'}),
                html.Span("= Alt_VSI", style={'marginRight': '20px'}),
                html.Span("C", style={'fontWeight': 'bold', 'color': '#1f77b4', 'marginRight': '5px'}),
                html.Span("= AI", style={'marginRight': '20px'}),
                html.Span("D", style={'fontWeight': 'bold', 'color': '#1f77b4', 'marginRight': '5px'}),
                html.Span("= TI_HSI", style={'marginRight': '20px'}),
                html.Span("E", style={'fontWeight': 'bold', 'color': '#1f77b4', 'marginRight': '5px'}),
                html.Span("= SSI", style={'marginRight': '20px'}),
                html.Span("F", style={'fontWeight': 'bold', 'color': '#1f77b4', 'marginRight': '5px'}),
                html.Span("= ASI", style={'marginRight': '20px'}),
                html.Span("G", style={'fontWeight': 'bold', 'color': '#1f77b4', 'marginRight': '5px'}),
                html.Span("= RPM", style={'marginRight': '20px'}),
                html.Span("H", style={'fontWeight': 'bold', 'color': '#1f77b4', 'marginRight': '5px'}),
                html.Span("= Window", style={'marginRight': '20px'}),
            ])
        ], style={'padding': '15px', 'backgroundColor': '#ffffff', 'borderRadius': '5px', 'marginBottom': '20px'}),
    ], style={'padding': '20px', 'marginBottom': '20px'}),


    # Parallel Coordinate Plot
    html.Div([
        html.H2("Parallel Coordinate Comparison: Successful vs Unsuccessful", style={'textAlign': 'center', 'marginBottom': '12px'}),
        html.Div([
            html.Div(dcc.Graph(id='pcp'), style={'width': '110%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ])
    ], style={'padding': '20px', 'marginBottom': '20px'}),

    # Saccade & Fixation Summary Metrics
    html.Div([
        html.H2("Saccade and Fixation Summary Metrics", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div([
            dcc.Graph(id='metrics-boxplots')
        ])
    ], style={'padding': '20px', 'marginBottom': '20px'}),

    # Legend/Glossary Section
    html.Div([
        html.H2("Metrics and Terminology Guide", style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#1f77b4'}),
        
        # AOI Definitions
        html.Div([
            html.H3("Areas of Interest (AOI)", style={'marginBottom': '10px'}),
            html.Div([
                html.Div([
                    html.Strong("A/No AOI"), " - Gaze did not fall within a pre-defined AOI"
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong("B/Alt_VSI"), " - Altimeter and Vertical Speed Indicator: Altitude and ascend/descend rate"
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong("C/AI"), " - Attitude Indicator: Aircraft pitch and roll"
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong("D/TI_HSI"), " - Turn Indicator and Horizontal Situation Indicator: Turn rate and heading"
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong("E/SSI"), " - Slip/Skid Indicator"
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong("F/ASI"), " - Airspeed Indicator"
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong("G/RPM"), " - Engine RPM"
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong("H/Window"), " - Outside view through windscreen"
                ], style={'marginBottom': '8px'}),
                
            ])
        ], style={'marginBottom': '25px', 'padding': '15px', 'borderRadius': '5px'}),
        
        # Eye Tracking Metrics
        html.Div([
            html.H3("Eye Tracking Metrics", style={'marginBottom': '10px'}),
            html.Div([
                html.Div([
                    html.Strong("Fixation"), " - When the eye remains relatively still on a target"
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong("Saccade"), " - Rapid eye movement between fixation points"
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong("Mean Fixation Duration"), " - Average time spent looking at each point where longer is deeper processing"
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong("Mean Saccade Amplitude"), " - Average distance in degrees of eye movements between fixations"
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong("Fixation-to-Saccade Ratio"), " - Proportion of time fixating vs moving eyes where higher is more focused)"
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong("Proportion/Duration"), " - Percentage of fixations or time spent looking at each AOI"
                ], style={'marginBottom': '8px'}),
            ])
        ], style={'marginBottom': '25px', 'padding': '15px', 'borderRadius': '5px'}),
        
        # Scanpath Metrics
        html.Div([
            html.H3("Scanpath Analysis Metrics", style={'marginBottom': '10px'}),
            html.Div([
                html.Div([
                    html.Strong("Scanpath Pattern"), " - Sequence of AOIs visited"
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong("Transition Matrix"), " - Shows probability of moving from one AOI to another"
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong("Stationary Entropy"), " - How evenly attention is distributed across AOIs where higher is less focused"
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong("Transition Entropy"), " - Randomness in eye movement patterns"
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong("Pattern Frequency"), " - Number of times a specific path pattern sequence appears"
                ], style={'marginBottom': '8px'}),
            ])
        ], style={'marginBottom': '25px', 'padding': '15px', 'borderRadius': '5px'}),
        
        # Performance Metrics
        html.Div([
            html.H3("Performance Classification", style={'marginBottom': '10px'}),
            html.Div([
                html.Div([
                    html.Strong("Approach Score"), " - Overall performance score for the ILS approach (0-1 scale)"
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong("Successful Pilot(Green)"), " - Approach Score ≥ 0.7"
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong("Unsuccessful Pilot(Red)"), " - Approach Score < 0.7"
                ], style={'marginBottom': '8px'}),
            ])
        ], style={'marginBottom': '25px', 'padding': '15px', 'borderRadius': '5px'}),
        
    ], style={'padding': '20px', 'marginBottom': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),

    create_instrument_guide(),
], style={'backgroundColor': COLORS['background']})

# Callback for Transition Heatmaps
@callback(
    Output('transition-heatmaps', 'figure'),
    Input('pilot-filter', 'data')
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
    Input('pilot-filter', 'data')
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
    Input('pilot-filter', 'data')
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
        'Average Pupil Size (mm)': 'average_pupil_size_of_both_eyes'
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


#callback for parallel plots
@callback(
    Output('pcp','figure'),
    Input('pilot-filter', 'data')

)
#creating paralle coordinate plots for successful and unsuccessful pilots
def update_pcp(selected_pilots):
    
    #filter df to only contain rows of selected pilots
    filtered_df = df[df['PID'].isin(selected_pilots)]

  
    #getting columns, and their plot names that will be used in PCP 
    pcp_columns = []
    column_names = []
    
    for col_name,col in PCP_COLUMNS.items():
        if col in df.columns:
            pcp_columns.append(col)
            column_names.append(col_name)


    #pcp_columns = [col for col in PCP_COLUMNS.values() if col in df.columns]
    #column_names = [col_name for col_name,col in PCP_COLUMNS if col in df.columns]

    #if no columns to include in pcp then return an empty figure
    if len(pcp_columns) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No PCP columns found in dataframe")
        return empty_fig, empty_fig
    

    # replace missing vals of #NULL! to nan
    filtered_df.replace('#NULL!',np.nan)
    
    #enocde pilot success columns
    filtered_df["pilot_success"] = filtered_df["pilot_success"].map({
        "Successful": 1,
        "Unsuccessful": 0
    })

    #convert all columns to numeric and pplies interpolation to columns with missing values
    filtered_df[pcp_columns] = filtered_df[pcp_columns].apply(pd.to_numeric, errors='coerce')
    filtered_df[pcp_columns] = filtered_df[pcp_columns].interpolate(method='linear', limit_direction='both')

    


  
    # Build dimension list for plotly
    dimensions = []
    for label, col in zip(column_names, pcp_columns):
        dimensions.append(
            dict(
                label=label,
                values=filtered_df[col]
            )
        )

    # Build plot
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=filtered_df["pilot_success"],
            colorscale=[
                [0, COLORS["unsuccessful"]],   # red = unsuccessful
                [1, COLORS["successful"]]      # green = successful
            ],
            showscale=True
        ),
        dimensions=dimensions
    ))

    fig.update_layout(
        title="Parallel Coordinates Plot: Successful vs Unsuccessful Pilots",
        height=700,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig

    '''def build_pcp(group_df,title):
        kept_cols = pcp_columns.copy()

        group_df = group_df.replace('#NULL!',n p.nan) #replace all #Null!(missing) values with Numpy nan represention

        group_df = group_df[kept_cols]

        #handling those same missing values via interpolation
        group_df = group_df[kept_cols].interpolate(method='polynomial',order = 3)


        #drop any missing values within columns used for pcp plot
        #group_df = group_df.dropna(subset=kept_cols,how='all')

        
        #if df is empty plot and Empty figure
        if len(group_df) == 0:
            empty = go.Figure()
            empty.update_layout(title=f"{title} (no data)")
            return empty


        
        
        
        #sample data from df if necessary(when df is >300 rows)
        group_df = _maybe_sample(group_df,max_rows=300)

        
        #normailze numeric columns
       # pcp_df = normalize_pcp_df(group_df,kept_cols)
        pcp_df = group_df

        #rename pcp_df col names
        rename_dict = {kept_cols[i]:column_names[i] for i in range(len(kept_cols))}
        pcp_df = pcp_df.rename(columns=rename_dict)


        fig = px.parallel_coordinates(
            pcp_df,
            dimensions=list(rename_dict.values()),
            color = 'Approach Score',          #map pcp line color to Pilot approach score
            #labels = {'Approach_Score':Approach_Score}
            range_color=(group_df['Approach_Score'].min(), group_df['Approach_Score'].max())

        )
        
        # reduce line width & add opacity-like effect by slightly adjusting color scale mapping
        #fig.update_traces(line=dict(colorscale='Viridis', showscale=True, cmin=group_df['Approach_Score'].min(), cmax=group_df['Approach_Score'].max(), width=1))

        fig.update_layout(
            title=title,
            height = 420,
            margin =dict(l=50,r=50,t=50,b=20)
        )
        return fig
    
    success_pcp_fig = build_pcp(success_df,"Succesful pilots")
    unsuccessful_pcp_fig = build_pcp(unsuccess_df,"Unsuccesful pilots")
    return success_pcp_fig,unsuccessful_pcp_fig'''







    




    




# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)

