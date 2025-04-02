import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
from collections import Counter
import plotly.graph_objects as go
from streamlit_echarts import st_echarts
from PIL import Image
from wordcloud import WordCloud
from collections import Counter
from pyecharts import options as opts
from pyecharts.charts import Bar
from streamlit_echarts import st_pyecharts
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
import random
import math
import warnings

@st.cache_resource
def check_eligibility(donor_data):
    """
    Détermine l'éligibilité d'un donneur de sang selon les critères médicaux.
    
    Retourne une des valeurs suivantes:
    - 'Eligible': Le donneur remplit tous les critères
    - 'Temporarily Not Eligible': Le donneur est temporairement non éligible
    - 'Not Eligible': Le donneur est définitivement non éligible
    """
    
    # Extraction des données pertinentes pour l'évaluation
    age = donor_data['age'].iloc[0]
    poids = donor_data['poids'].iloc[0]
    date_jour = datetime.now().date()
    
    # Initialisation des variables pour les raisons d'inéligibilité
    permanent_reasons = []
    temporary_reasons = []
    
    # Vérification des critères de base (âge et poids)
    if age < 18 or age > 70:
        permanent_reasons.append(f"Âge non conforme ({age} ans)")
    
    if poids < 50:
        permanent_reasons.append(f"Poids insuffisant ({poids} kg)")
    
    # Vérification du taux d'hémoglobine
    if 'taux_dhemoglobine' in donor_data.columns and not pd.isna(donor_data['taux_dhemoglobine'].iloc[0]):
        taux_hb = donor_data['taux_dhemoglobine'].iloc[0]
        genre = donor_data['genre'].iloc[0]
        
        # Seuils généralement admis (peuvent varier selon les pays)
        if (genre.lower() == 'homme' or genre.lower() == 'masculin') and taux_hb < 13:
            temporary_reasons.append(f"Taux d'hémoglobine bas ({taux_hb} g/dL)")
        elif (genre.lower() == 'femme' or genre.lower() == 'féminin') and taux_hb < 12:
            temporary_reasons.append(f"Taux d'hémoglobine bas ({taux_hb} g/dL)")
    
    # Vérification du dernier don
    if donor_data['deja_donne_sang'].iloc[0] == 'Yes' and donor_data['date_dernier_don'].iloc[0]:
        try:
            date_dernier_don = donor_data['date_dernier_don'].iloc[0]  # Assuming this is already a datetime.date object

            delai_depuis_dernier_don = (date_jour - date_dernier_don).days
            
            if delai_depuis_dernier_don < 56:  # 8 weeks = 56 days
                temporary_reasons.append(f"Dernier don trop récent ({delai_depuis_dernier_don} jours)")
        except ValueError:
            # Si la date n'est pas dans le bon format, on ne peut pas vérifier
            pass
    
    # Vérification des contre-indications permanentes
    permanent_conditions = [
        ('drepanocytaire', "Drépanocytose"),
        ('porteur_hiv_hbs_hcv', "Porteur VIH, hépatite B ou C"),
        ('diabetique', "Diabète"),
        ('cardiaque', "Maladie cardiaque")
    ]
    
    for col, reason in permanent_conditions:
        if col in donor_data.columns and donor_data[col].iloc[0] == 'Yes':
            permanent_reasons.append(reason)
    
    # Vérification des contre-indications temporaires
    temporary_conditions = [
        ('est_sous_anti_biotherapie', "Sous antibiothérapie"),
        ('taux_dhemoglobine_bas', "Taux d'hémoglobine bas"),
        ('date_dernier_don_3_mois', "Don récent (moins de 3 mois)"),
        ('ist_recente', "IST récente"),
        ('ddr_incorrecte', "DDR incorrecte"),
        ('allaitement', "Allaitement"),
        ('accouchement_6mois', "Accouchement récent (moins de 6 mois)"),
        ('interruption_grossesse', "Interruption de grossesse récente"),
        ('enceinte', "Grossesse en cours"),
        ('antecedent_transfusion', "Transfusion récente"),
        ('opere', "Opération récente"),
        ('tatoue', "Tatouage récent"),
        ('scarifie', "Scarification récente"),
        ('hypertendus', "Hypertension non contrôlée"),
        ('asthmatiques', "Crise d'asthme récente")
    ]
    
    for col, reason in temporary_conditions:
        if col in donor_data.columns and donor_data[col].iloc[0] == 'Yes':
            temporary_reasons.append(reason)
    
    # Détermination de l'éligibilité finale
    if permanent_reasons:
        eligibility = 'Définitivement non-eligible'
        reasons = permanent_reasons
    elif temporary_reasons:
        eligibility = 'Temporairement Non-eligible'
        reasons = temporary_reasons
    else:
        eligibility = 'Eligible'
        reasons = []
    
    return {
        'eligibility': eligibility,
        'reasons': reasons
    }


@st.cache_resource
def message(text):
    st.markdown(
        f"""
        <div class="celebration-container">
            <h1 class="celebration-text">{text}</h1>
        </div>
        
        <style>
            .celebration-container {{
                text-align: center;
                padding: 40px;
                position: relative;
                overflow: hidden;
                height: 300px;
            }}

            .celebration-text {{
                font-size: 4em;
                background: linear-gradient(45deg, #ff0000, #ff7700, #ffff00, #00ff00, #0000ff, #8b00ff);
                background-size: 600% 600%;
                animation: gradient 6s ease infinite, bounce 2s ease infinite;
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }}

            .fireworks {{
                position: absolute;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background-color: #ffcc00;
                animation: fireworks 2s ease infinite;
                opacity: 0;
                left: 50%;
                bottom: 0;
            }}

            @keyframes gradient {{
                0% {{ background-position: 0% 50%; }}
                50% {{ background-position: 100% 50%; }}
                100% {{ background-position: 0% 50%; }}
            }}

            @keyframes bounce {{
                0%, 100% {{ transform: translateY(0); }}
                50% {{ transform: translateY(-20px); }}
            }}

            @keyframes fireworks {{
                0% {{
                    transform: translate(50%, 100%) scale(1);
                    opacity: 1;
                }}
                25% {{
                    opacity: 1;
                }}
                100% {{ 
                    transform: translate(var(--x), var(--y)) scale(3);
                    opacity: 0;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

@st.cache_resource(show_spinner=False)
def compute_age_distribution(age_data):
    """
    Computes age distribution for an ECharts bar chart.
    
    Parameters:
    age_data (pd.Series or pd.DataFrame): Series containing age values
    
    Returns:
    dict: ECharts option dictionary for plotting
    """
    # Convert input to hashable format for proper caching
    if isinstance(age_data, pd.DataFrame):
        if len(age_data.columns) == 1:
            age_data = age_data.iloc[:, 0].values.tolist()
        else:
            age_data = str(age_data)  # Convert to string representation for caching
    elif isinstance(age_data, pd.Series):
        age_data = age_data.values.tolist()

    age_data_processed = pd.Series(age_data) if not isinstance(age_data, str) else pd.Series([])

    # Define age categories
    bins = [0, 17, 25, 35, 50, float('inf')]
    labels = ["Moins de 18 ans", "18-25 ans", "26-35 ans", "36-50 ans", "Plus de 50 ans"]

    # Categorize ages
    age_categories = pd.cut(age_data_processed, bins=bins, labels=labels, right=True)

    # Count occurrences in each category
    age_frequencies = age_categories.value_counts().reindex(labels, fill_value=0).tolist()

    # Define the ECharts option
    option = {
        "tooltip": {"trigger": "axis", "formatter": "{b}: {c}"},
        "xAxis": {
            "type": "category",
            "data": labels,
            "name": "Age Group",
            "nameLocation": "middle",
            "nameGap": 35,
            "axisLabel": {"rotate": 0}
        },
        "yAxis": {"name": "Number of People"},
        "series": [
            {
                "name": "Count",
                "type": "bar",
                "data": age_frequencies,
                "barWidth": "90%",
                "itemStyle": {
                    "color": "rgba(99, 110, 250, 0.8)",
                    "borderColor": "rgba(50, 50, 150, 1)",
                    "borderWidth": 1,
                    "borderRadius": [3, 3, 0, 0]
                },
                "label": {"show": True, "position": "top", "formatter": "{c}"}
            }
        ],
        "grid": {"top": "10%", "bottom": "15%", "left": "8%", "right": "5%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}, "dataZoom": {}, "restore": {}}, "right": "10%"}
    }
    
    return option




@st.cache_resource
def prepare_radar_chart_options(data, categories, title=""):
    """
    Prépare les options pour le graphique radar et les retourne sans les afficher.
    Cette fonction peut être mise en cache en toute sécurité.
    """
    # Define radar chart indicators with auto-adjusted max values
    max_value = max([max(values) for values in data.values()]) * 1.2
    indicators = [{"name": cat, "max": max_value} for cat in categories]

    # Prepare data for ECharts
    series_data = [
        {
            "name": key,
            "value": value,
            "areaStyle": {"opacity": 0.5},  # Add filled color
            "itemStyle": {"color": "red"} 
        }
        for key, value in data.items()
    ]

    # Define ECharts options
    option = {
       "title": {"text": title, "textStyle": {"fontSize": 18, "color": "#333"}},
        "tooltip": {"trigger": "item"},
        "legend": {"data": list(data.keys()), "left": "14%"},
        "radar": {
            "indicator": indicators,
            "shape": "circle",  # Make radar circular
            "splitNumber": 5,  # Reduced number of grid levels for more spacing
            "axisName": {"color": "#333"},  # Category label color
            "splitLine": {
                "lineStyle": {
                    "type": "solid",  # Changed to dashed for better visibility
                    "width": 2,
                    "dashOffset": 0,  # Add dash offset
                    "cap": "round"
                }
            },  # Grid lines
            "splitArea": {
                "show": False,
                "areaStyle": {
                    "color": ["red", "white"]
                }  # Alternating background colors
            },
        },
        "series": [
            {
                "name": title,
                "type": "radar",
                "data": series_data,
                "tooltip": {
                    "show": True
                }
            }
        ]
    }
    
    return option

def plot_radar_chart(data, categories, title="", height="400px"):
    """
    Fonction de rendu qui affiche le graphique en utilisant les options préparées
    par la fonction mise en cache.
    """
    # Obtenir les options depuis la fonction mise en cache
    options = prepare_radar_chart_options(data, categories, title)
    
    # Rendre le graphique (cette partie ne peut pas être mise en cache)
    st_echarts(options=options, height=height, width="100%")
df = pd.read_excel('last.xlsx')
#df['Date de remplissage de la fiche'] = pd.to_datetime(df['Date de remplissage de la fiche'])
# m= df['Date de remplissage de la fiche'].dt.month
# month_counts = m.value_counts().sort_index()
# categories = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
# data = {"Nombre d'occurences": month_counts.reindex(range(1, 13), fill_value=0).tolist()}
# plot_radar_chart(data, categories)


from collections import Counter
import streamlit as st
from streamlit_echarts import st_echarts

# Fonctions de préparation (mises en cache)
@st.cache_resource
def prepare_frequency_pie_options_wl(data_list, lege):
    """
    Prépare les options pour le graphique pie et les retourne sans les afficher.
    """
    # Count the frequency of each value
    frequency = Counter(data_list)
    
    # Convert to the format needed for ECharts and sort by value in descending order
    pie_data = [{"name": str(key), "value": value} for key, value in sorted(frequency.items(), key=lambda x: x[1], reverse=True)]
    
    # Define pie chart options
    pie_options = {
        "tooltip": {
            "trigger": "item",
            "formatter": "{b} : <br/>{c} occurences ({d}%)"
        },
        "legend": {
            "show": True,
            "orient": "vertical",
            "left": "70%",
            "top": "60%"
        },
        "series": [
            {
                "type": "pie",
                "radius": "95%",
                "center": ["52%", "40%"],  # Position the pie [x, y]
                "data": pie_data,
                "roseType": "radius",
                "label": {
                    "show": False  # Remove text labels on pie segments
                },
                "labelLine": {
                    "show": False  # Hide label lines
                },
                "itemStyle": {
                    "shadowBlur": 30,
                    "shadowColor": "rgba(0, 0, 0, 0.5)",
                },
                "animationType": "scale",
                "animationEasing": "elasticOut",
            }
        ],
        # Set a colorful palette for the pie chart
        "color": ["#FA668F", "#B22222", "#DC143C", "#EC8282", "#FFB6B6", "#FFD1D1"]
    }
    
    return pie_options

@st.cache_resource
def prepare_frequency_pie_options(data_list, legend_left="70%", legend_top="50%"):
    """
    Prépare les options pour le graphique pie standard et les retourne sans les afficher.
    """
    # Count the frequency of each value
    frequency = Counter(data_list)
    
    # Convert to the format needed for ECharts and sort by value in descending order
    pie_data = [{"name": str(key), "value": value} for key, value in sorted(frequency.items(), key=lambda x: x[1], reverse=True)]
    
    # Define pie chart options
    pie_options = {
        "tooltip": {
            "trigger": "item",
            "formatter": "<span style='font-size:11px; font-weight:bold;'>{b} :</span> <br/>"
                     "<span style='font-size:12px;'>{c} occurrences ({d}%)</span>"
        },
        "legend": {
            "show": True,
            "orient": "vertical",
            "left": legend_left,
            "top": legend_top,
            "textStyle": {
                "fontSize": 11, # Optional: Makes the text bold
                "color": "#333"  # Optional: Sets the text color
            }
        },
        "series": [
            {
                "type": "pie",
                "radius": "105%",
                "center": ["60%", "55%"],  # Position the pie [x, y]
                "data": pie_data,
                "roseType": "radius",
                "label": {
                    "show": False  # Remove text labels on pie segments
                },
                "labelLine": {
                    "show": False # Hide label lines
                },
                "itemStyle": {
                    "shadowBlur": 30,
                    "shadowColor": "rgba(0, 0, 0, 0.5)",
                },
                "animationType": "scale",
                "animationEasing": "elasticOut",
            }
        ],
        # Set a colorful palette for the pie chart
        "color": ["#FA668F",   "SlateBlue", "SandyBrown", "pink", "cyan","green"]
    }
    
    return pie_options

@st.cache_resource
def prepare_frequency_pieh_options(data_list, legend_left="70%", legend_top="85%"):
    """
    Prépare les options pour le graphique pie avec trou (donut) et les retourne sans les afficher.
    """
    # Count the frequency of each value
    frequency = Counter(data_list)

    # Convert to the format needed for ECharts and sort by value in descending order
    pie_data = [{"name": str(key), "value": value} for key, value in sorted(frequency.items(), key=lambda x: x[1], reverse=True)]

    # Define pie chart options
    pie_options = {
        "tooltip": {
            "trigger": "item",
            "formatter": "{b} : <br/>{c} occurrences ({d}%)"
        },
        "legend": {
            "show": True,
            "orient": "vertical",
            "left": legend_left,
            "top": legend_top
        },
        "series": [
            {
                "type": "pie",
                "radius": ["25%", "70%"],  # Creates a donut effect
                "center": ["50%", "48%"],  # Position of the pie chart
                "data": pie_data,
                "roseType": "radius",
                "label": {
                    "show": True  # Show labels on pie segments
                },
                "labelLine": {
                    "show": True  # Show label lines
                },
                "itemStyle": {
                    "shadowBlur": 30,
                    "shadowColor": "rgba(0, 0, 0, 0.5)",
                },
                "animationType": "scale",
                "animationEasing": "elasticOut",
            }
        ],
        # Set a colorful palette for the pie chart
        "color": ["#FA668F",  "SlateBlue", "tomato","SandyBrown", "pink", "#FF3434","cyan","skyblue"]
    }

    return pie_options

# Fonctions de rendu (non mises en cache)
def render_frequency_pie_wl(data_list, lege):
    """
    Fonction de rendu pour graphique pie type waterloo
    """
    options = prepare_frequency_pie_options_wl(data_list, lege)
    st_echarts(options=options, height="400px")

def render_frequency_pie(data_list, legend_left="70%", legend_top="50%"):
    """
    Fonction de rendu pour graphique pie standard
    """
    options = prepare_frequency_pie_options(data_list, legend_left, legend_top)
    st_echarts(options=options, height="400px")

def render_frequency_pieh(data_list, legend_left="70%", legend_top="65%"):
    """
    Fonction de rendu pour graphique pie avec trou (donut)
    """
    options = prepare_frequency_pieh_options(data_list, legend_left, legend_top)
    st_echarts(options=options, height="350px")
#render_frequency_pie2(pd.read_excel("last.xlsx")["Niveau_d'etude"])


@st.cache_data
def count_frequencies(data_list):
    """
    Count the frequency of each value in the data list.

    Parameters:
    - data_list: List of values to count frequencies.

    Returns:
    - List of dictionaries with names and values for ECharts.
    """
    frequency = Counter(data_list)
    return [{"name": str(key), "value": value} for key, value in sorted(frequency.items(), key=lambda x: x[1], reverse=True)]

def render_frequency_pie2(pie_data, legend_left="23%", legend_top="80%"):
    """
    Render a pie chart with a hole (donut chart) using ECharts in Streamlit.

    Parameters:
    - pie_data: List of dictionaries with names and values for ECharts.
    - legend_left: Horizontal position of the legend (default: "75%").
    - legend_top: Vertical position of the legend (default: "50%").
    """
    # Define pie chart options
    pie_options = {
        "tooltip": {
            "trigger": "item",
            "formatter": "{b} : <br/>{c} Values<br/>({d}%)"
        },
        "legend": {
            "show": True,
            "orient": "vertical",
            "left": legend_left,
            "top": legend_top
        },
        "series": [
            {
                "type": "pie",
                "radius": ["25%", "70%"],  # Creates a donut effect
                "center": ["50%", "45%"],  # Position of the pie chart
                "data": pie_data,
                "roseType": "radius",
                "label": {
                    "show": True  # Show labels on pie segments
                },
                "labelLine": {
                    "show": True  # Show label lines
                },
                "itemStyle": {
                    "shadowBlur": 30,
                    "shadowColor": "rgba(0, 0, 0, 0.5)",
                },
                "animationType": "scale",
                "animationEasing": "elasticOut",
            }
        ],
        # Set a colorful palette for the pie chart
        "color": ["#FA668F",  "SlateBlue", "tomato","SandyBrown", "pink", "#FF3434","cyan","skyblue"]
    }

    # Render the chart in Streamlit
    st_echarts(options=pie_options, height="300px")

@st.cache_resource
def plot_age_pyramid(df, age_col='Age', gender_col='Genre_', bin_size=4, height=600):
    """
    Plots an age pyramid using Plotly in Streamlit.

    Parameters:
    df (DataFrame): The dataset containing age and gender columns.
    age_col (str): Column name for age data.
    gender_col (str): Column name for gender data.
    bin_size (int): Size of age bins.
    height (int): Height of the figure in pixels.

    Returns:
    None (Displays the plot in Streamlit).
    """
    # Define age bins
    bin_edges = np.arange(df[age_col].min(), df[age_col].max() + bin_size, bin_size)
    bin_labels = [f"{int(age)}-{int(age + bin_size - 1)}" for age in bin_edges[:-1]]

    # Assign age groups
    df['AgeGroup'] = pd.cut(df[age_col], bins=bin_edges, labels=bin_labels, right=False)

    # Group by age group and gender
    hommes = df[df[gender_col] == 'Homme'].groupby('AgeGroup').size().reset_index(name='Count')
    femmes = df[df[gender_col] == 'Femme'].groupby('AgeGroup').size().reset_index(name='Count')

    # Assign gender labels
    hommes['Genre'] = 'Hommes'
    femmes['Genre'] = 'Femmes'

    # Concatenate data
    data = pd.concat([hommes, femmes])

    # Invert female values for visualization
    data.loc[data['Genre'] == 'Femmes', 'Count'] *= -1

    # Create the age pyramid plot
    fig = px.bar(data, 
                 y='AgeGroup', 
                 x='Count', 
                 color='Genre', 
                 orientation='h',
                 labels={'Count': 'Number of people ', 'AgeGroup': 'Âge'},
                 color_discrete_map={'Hommes': '#DC143C', 'Femmes':  'SlateBlue'})

    # Customize layout
    fig.update_layout(
        height=height,  # Set the figure height
        yaxis=dict(showgrid=True, categoryorder="category ascending"),
        xaxis=dict(title='Nombre de personnes', showgrid=True),
        plot_bgcolor='rgba(0,0,0,0)',
        barmode='overlay',
        bargap=0.1,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

@st.cache_resource
def number_line(df):

    nombre_total_donneurs = df.shape[0]

    a = df['Date de remplissage de la fiche'].dropna()

    # Convertir en datetime avec gestion des erreurs
    dates = pd.to_datetime(a, format="%m/%d/%Y %H:%M", errors='coerce').dropna()

    a = dates.value_counts()
    a.sort_index(inplace=True)
    #a = a[(a.index >= "2000-01-01") & (a.index <= "2022-12-31")]

    # Make the line thinner (reduced from 3 to 1.5)
    fig = px.line(a, x=a.index, y=a.values, labels={'x': 'Années', 'y': 'Nombre d\'occurrences'}, line_shape='spline')
    fig.update_traces(line_color='#f83e8c', line_width=1.5, 
                      fill='tozeroy', fillcolor='rgba(232, 62, 140, 0.1)') 
    
    # Enhance the layout
    fig.update_layout(
        height=400, 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.05,
                bgcolor='rgba(0,0,0,0.05)',
            ),
            rangeslider_bordercolor='#aaaaaa', 
            rangeslider_borderwidth=1,
            showgrid=True,
            gridcolor='rgba(211,211,211,0.3)',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(211,211,211,0.3)',
        ),
    )

    st.plotly_chart(fig, use_container_width=True)
@st.cache_data
def generate_wordcloud_and_barchart(df):
    # Extract words related to donation reasons
    words = " ".join(df["Raison_indisponibilité_fusionnée"].dropna().astype(str).tolist())

    # Compute word frequencies
    word_list = words.split()
    word_freq = {}
    for word in word_list:
        word_freq[word] = word_freq.get(word, 0) + np.random.randint(10, 50)  # Simulated frequencies

    # Boost specific keywords
    key_words = ["sang", "donneur", "vie", "santé", "sauver", "solidarité", "don"]
    for word in key_words:
        if word in word_freq:
            word_freq[word] *= 3  # Increase importance

    # Define a good color palette (blood and health theme)
    colors = ["#8B0000", "#B22222", "#DC143C", "#E9967A", "#FF6347", "#FFA07A"]

    # Define a color function for the WordCloud
    
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return np.random.choice(colors)

    # Create a circular mask
    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    mask = 255 * mask.astype(int)

    # Generate the word cloud
    wc = WordCloud(
        background_color="white",
        max_words=150,
        mask=mask,
        color_func=color_func,  # Use custom colors
        max_font_size=80,
        random_state=42,
        width=800,
        height=800,
        contour_width=1.5,
        contour_color='#8B0000'  # Dark red contour
    ).generate_from_frequencies(word_freq)

    # Convert word cloud image to base64
    img = wc.to_image()
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Create a Plotly figure with the word cloud image
    fig = go.Figure()

    # Add image as a background
    fig.add_layout_image(
        dict(
            source='data:image/png;base64,' + img_str,
            x=0,
            y=1,
            xref="paper",
           
            sizex=1,
            sizey=1,
        )
    )

    # Configure layout for better visibility
    fig.update_layout(
        width=400,
        height=300,
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis={'showgrid': False, 'showticklabels': False, 'zeroline': False},
        yaxis={'showgrid': False, 'showticklabels': False, 'zeroline': False},
        margin=dict(l=140, r=0, t=0, b=0, pad=0)  # Remove all margins and padding

       
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Example usage
# df = pd.DataFrame({'Raison_indisponibilité_fusionnée': ['fatigue; maladie; voyage', 'travail; fatigue; grippe']})
# generate_interactive_wordcloud(df)


@st.cache_resource
def circle(df):
    # Définir les colonnes
    cols = ["Classe_Age", "Niveau_d'etude", 'Genre_', 'Religion_Catégorie', 
        "Situation_Matrimoniale_(SM)",  "categories"]

    # Extraire les modalités et leurs fréquences
    data = df[cols]
    groupes = []
    for col in cols:
        groupes.append(data[col].value_counts().index.tolist())
    groupes = sum(groupes, [])  # Aplatir la liste

    # Ajouter des sauts de ligne dans les étiquettes longues
    groupes = [label.replace(" ", "<br>") for label in groupes]  # Exemple : remplacer les espaces par <br>

    # Calculer les totaux par modalité
    modal = []
    for col in cols: 
        modal.append(data[col].value_counts().tolist())
    total = sum(modal, [])  # Aplatir la liste

    # Calculer les pourcentages par éligibilité pour chaque modalité
    pourcentages = []

    for col in cols:
        modalites = df[col].unique()
        for mod in modalites:
            pourcentages_modalite = []
            df_modalite = df[df[col] == mod]
            total_modalite = len(df_modalite)
            
            # Ne pas continuer si aucune donnée
            if total_modalite == 0:
                continue
                
            for eligibilite in ['Eligible', 'Temporairement Non-eligible', 'Définitivement non-eligible']:
                # S'assurer que la colonne existe
                if 'ÉLIGIBILITÉ_AU_DON.' in df.columns:
                    count = len(df_modalite[df_modalite['ÉLIGIBILITÉ_AU_DON.'] == eligibilite])
                    percentage = round((count / total_modalite) * 100)
                    pourcentages_modalite.append(percentage)
                else:
                    pourcentages_modalite.append(0)  # Valeur par défaut si la colonne n'existe pas
                    
            pourcentages.append(pourcentages_modalite)

    # Vérifier que les listes ont la même longueur
    min_length = min(len(groupes), len(total), len(pourcentages))
    groupes = groupes[:min_length]
    total = total[:min_length]
    pourcentages = pourcentages[:min_length]

    # Calculer les valeurs pour chaque catégorie d'éligibilité
    def_non_eligibles = [total[i] * pourcentages[i][0] / 100 for i in range(len(total))]
    temp_non_eligibles = [total[i] * pourcentages[i][1] / 100 for i in range(len(total))]
    eligibles = [total[i] * pourcentages[i][2] / 100 for i in range(len(total))]

    # Créer des angles pour chaque groupe
    theta = np.linspace(0, 2*np.pi, len(groupes), endpoint=False)
    # Ajuster l'ordre pour que le graphique commence en haut
    theta = np.roll(theta, len(theta)//4)
    groupes_roll = np.roll(groupes, len(groupes)//4).tolist()
    total_roll = np.roll(total, len(total)//4).tolist()
    def_non_eligibles_roll = np.roll(def_non_eligibles, len(eligibles)//4).tolist()
    temp_non_eligibles_roll = np.roll(temp_non_eligibles, len(temp_non_eligibles)//4).tolist()
    eligibles_roll = np.roll(eligibles, len(def_non_eligibles)//4).tolist()

    # Créer le graphique
    fig = go.Figure()

    # Ajouter "Définitivement non-éligibles" (gris clair)
    fig.add_trace(go.Barpolar(
        r=[def_non_eligibles_roll[i] + temp_non_eligibles_roll[i] + eligibles_roll[i] for i in range(len(groupes_roll))],
        theta=groupes_roll,
        name="Éligibles",
        marker_color="lightgreen", 
        width=1
    ))

    # Ajouter "Temporairement non-éligibles" (bleu)
    fig.add_trace(go.Barpolar(
        r=[temp_non_eligibles_roll[i] + eligibles_roll[i] for i in range(len(groupes_roll))],
        theta=groupes_roll,
        name="Temporairement non-éligibles",
        marker_color="#B22222", 
        width=1
    ))

    # Ajouter "Éligibles" (vert)
    fig.add_trace(go.Barpolar(
        r=eligibles_roll,
        theta=groupes_roll,
        name="Définitivement non-éligibles",
        marker_color="blue", 
        width=1
    ))

    # Ajouter du texte pour les totaux uniquement (sans les étiquettes)
    for i in range(len(groupes_roll)):
        fig.add_trace(go.Scatterpolar(
            r=[1.1*max(total_roll)],
            theta=[groupes_roll[i]],
            text=[f"{total_roll[i]}"],  # Suppression de {groupes_roll[i]}<br>
            mode="text",
            showlegend=False,
            textfont=dict(size=8)
        ))

    # Configurer le layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(total_roll)*1.2],
                tickvals=[0, max(total_roll)*0.2, max(total_roll)*0.4, max(total_roll)*0.6, max(total_roll)*0.8, max(total_roll)],
                ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
                tickfont=dict(size=11),
            gridcolor="rgba(20, 0, 0, 0.25)",  # ✅ Correct RGBA format
            griddash="dot"
            ),
            angularaxis=dict(
                direction="clockwise",
                tickfont=dict(size=12),
                gridcolor="rgba(20, 0, 0, 0.25)",  # ✅ Correct RGBA format
            griddash="dot"  # Ajuster la taille de la police des étiquettes
            )
        ),
        legend=dict(
            title="Type de donneurs:",
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        width=900,
        height=900
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)

@st.cache_data
def process_donation_dates(df):
    """
    Process donation dates from the dataframe.
    Returns a DataFrame with cleaned dates and weekday information.
    
    Args:
        df: Input DataFrame containing donation data
    
    Returns:
        DataFrame with processed dates
    """
    # Extract and clean dates
    dates_raw = df['Date de remplissage de la fiche'].dropna()

    # Convert to datetime with error handling
    dates = pd.to_datetime(dates_raw, format="%m/%d/%Y %H:%M", errors='coerce').dropna()

    # Create a DataFrame with dates
    dates_df = pd.DataFrame({'Date': dates})
    dates_df['Year'] = dates_df['Date'].dt.year
    dates_df['Weekday'] = dates_df['Date'].dt.day_name()  # Get day name (Monday, Tuesday, etc.)
    
    return dates_df

@st.cache_data
def analyze_weekday_donations(dates_df, year_filter=(2019, 2020)):
    """
    Analyze donations by weekday for specific years.
    
    Args:
        dates_df: DataFrame with processed dates
        year_filter: Tuple of years to filter by
    
    Returns:
        Tuple containing:
        - Weekday counts
        - Weekday order
        - Max donation day and count
    """
    # Define the order of days for sorting
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Filter for specified years
    filtered_data = dates_df[dates_df['Year'].isin(year_filter)]
    
    # Count donations by weekday
    weekday_counts = filtered_data['Weekday'].value_counts().reindex(weekday_order, fill_value=0)
    
    # Find the day with maximum donations
    max_count = weekday_counts.max()
    max_day = weekday_counts.idxmax() if not weekday_counts.empty else "No data"
    
    return weekday_counts, weekday_order, (max_day, max_count)

@st.cache_data
def prepare_chart_data(weekday_counts, weekday_order):
    """
    Prepare data for the scatter chart.
    
    Args:
        weekday_counts: Series with counts by weekday
        weekday_order: List with weekday order
    
    Returns:
        List of formatted data points with sizing
    """
    # Calculate the position for each day on x-axis (0 to 6)
    weekday_positions = {day: i for i, day in enumerate(weekday_order)}
    
    # Format the data for the chart
    formatted_data = []
    for day, count in weekday_counts.items():
        formatted_data.append([
            weekday_positions[day],  # Position on x-axis (0 to 6)
            count,                   # Number of donations (y-axis)
            count,                   # For symbol size (same as count)
            day,                     # Day name as label
            2019                     # Year (hardcoded as 2019 for now)
        ])
    
    # Calculate the symbol size for each point
    data_with_size = []
    for point in formatted_data:
        size = math.sqrt(point[2]) * 5  # Scaling factor for bubble size
        data_with_size.append({
            "value": point,
            "symbolSize": size,
            "name": point[3],
            "label": {
                "position": "top",
                "formatter": "{c}",
                "fontSize": 12,
                "color": "rgb(204, 46, 72)"
            }
        })
    
    return data_with_size

@st.cache_data
def create_chart_options(data_with_size, weekday_order):
    """
    Create the ECharts options for the scatter chart.
    
    Args:
        data_with_size: List of formatted data points
        weekday_order: List with weekday order
    
    Returns:
        Dict with chart options
    """
    option = {
        "backgroundColor": {
            "type": "radialGradient",
            "x": 0.3,
            "y": 0.3,
            "r": 0.8,
            "colorStops": [
                {
                    "offset": 0,
                    "color": "#f7f8fa"
                },
                {
                    "offset": 1,
                    "color": "#cdd0d5"
                }
            ]
        },
        "grid": {
            "left": "8%",
            "top": "15%",
            "right": "8%",
            "bottom": "12%"
        },
        "tooltip": {},
        "xAxis": {
            "type": "category",
            "data": weekday_order,
            "name": "Jour de la semaine",
            "nameLocation": "middle",
            "nameGap": 30,
            "axisLine": {
                "lineStyle": {
                    "color": "#999"
                }
            },
            "axisLabel": {
                "rotate": 0,
                "fontSize": 10
            }
        },
        "yAxis": {
            "type": "value",
            "name": "Nombre de dons",
            "nameLocation": "middle",
            "nameGap": 20,
            "axisLine": {
                "lineStyle": {
                    "color": "#999"
                }
            },
            "splitLine": {
                "lineStyle": {
                    "type": "dashed"
                }
            }
        },
        "series": [
            {
                "data": data_with_size,
                "type": "scatter",
                "emphasis": {
                    "focus": "series",
                    "itemStyle": {
                        "shadowBlur": 20,
                        "shadowColor": "rgba(120, 36, 50, 0.7)"
                    }
                },
                "itemStyle": {
                    "shadowBlur": 10,
                    "shadowColor": "rgba(120, 36, 50, 0.5)",
                    "shadowOffsetY": 5,
                    "color": {
                        "type": "radialGradient",
                        "x": 0.4,
                        "y": 0.3,
                        "r": 1,
                        "colorStops": [
                            {
                                "offset": 0,
                                "color": "rgb(251, 118, 123)"
                            },
                            {
                                "offset": 1,
                                "color": "rgb(204, 46, 72)"
                            }
                        ]
                    }
                }
            }
        ]
    }
    
    return option

def jour(df, height="200px"):
    """
    Main function to display the weekly donation pattern chart.
    
    Args:
        df: Input DataFrame containing donation data
        height: Chart height
    """
    # Process the data (cached)
    dates_df = process_donation_dates(df)
    
    # Analyze weekday donations (cached)
    weekday_counts, weekday_order, (max_day, max_count) = analyze_weekday_donations(dates_df)
    
    # Prepare chart data (cached)
    data_with_size = prepare_chart_data(weekday_counts, weekday_order)
    
    # Create chart options (cached)
    chart_options = create_chart_options(data_with_size, weekday_order)
    
      
    nombre_total_donneurs = df.shape[0]

    # Extract and clean dates
    dates_raw = df['Date de remplissage de la fiche'].dropna()

    # Convert to datetime with error handling
    dates = pd.to_datetime(dates_raw, format="%m/%d/%Y %H:%M", errors='coerce').dropna()

    # Create a DataFrame with dates
    dates_df = pd.DataFrame({'Date': dates})
    dates_df['Year'] = dates_df['Date'].dt.year
    dates_df['Weekday'] = dates_df['Date'].dt.day_name()  # Get day name (Monday, Tuesday, etc.)

    # Filter for 2019 and 2020
    data_2019 = dates_df[(dates_df['Year'] == 2019) | (dates_df['Year'] == 2020)]

    # Define the order of days for sorting
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Count donations by weekday for each year
    counts_2019 = data_2019['Weekday'].value_counts().reindex(weekday_order, fill_value=0)


    # Prepare data in the requested format
    # Using [index, count, total_for_sizing, 'Label', year]
    data_2019_formatted = []


    # Calculate the position for each day on x-axis (0 to 6)
    weekday_positions = {day: i for i, day in enumerate(weekday_order)}

    # For 2019
    for day, count in counts_2019.items():
        data_2019_formatted.append([
            weekday_positions[day],  # Position on x-axis (0 to 6)
            count,                   # Number of donations (y-axis)
            count,                   # For symbol size (same as count)
            day,                     # Day name as label
            2019                     # Year
        ])


    # Set up the data in the required structure
    data = [
        data_2019_formatted,  # Data for 2019
    ]

    # Pre-calculate symbol sizes for each data point
    data_2019_with_size = []


    # Find max values to determine which point should have a label
    max_count_2019 = 0
    max_day_2019 = ""
    ""

    # Calculate the size for each point and find max values
    for point in data[0]:
        size = math.sqrt(point[2]) * 5  # Scaling factor for bubble size
        if point[1] > max_count_2019:
            max_count_2019 = point[1]
            max_day_2019 = point[3]
        
        data_2019_with_size.append({
            "value": point,
            "symbolSize": size,
            "name": point[3]
        })



    # Add labels to all points
    for i, point_data in enumerate(data_2019_with_size):
        point_data["label"] = {
            #"show": True,
            "position": "top",
            "formatter": "{c}",  # Show both day name and count
            "fontSize": 12,
            "color": "rgb(204, 46, 72)"
        }


    # Define the option dictionary
    option = {
        "backgroundColor": {
            "type": "radialGradient",
            "x": 0.3,
            "y": 0.3,
            "r": 0.8,
            "colorStops": [
                {
                    "offset": 0,
                    "color": "#f7f8fa"
                },
                {
                    "offset": 1,
                    "color": "#cdd0d5"
                }
            ]
        },

        "grid": {
            "left": "8%",
            "top": "15%",
            "right": "8%",
            "bottom": "12%"
        },
        "tooltip": {
            #"formatter": "{c}: {b} donations"  # Corrigé: utiliser un string au lieu d'une fonction JavaScript
        },
        "xAxis": {
            "type": "category",
            "data": weekday_order,
            "name": "Jour de la semaine",
            "nameLocation": "middle",
            "nameGap": 30,
            "axisLine": {
                "lineStyle": {
                    "color": "#999"
                }
            },
            "axisLabel": {
                "rotate": 0,
                "fontSize": 10
            }
        },
        "yAxis": {
            "type": "value",
            "name": "Nombre de dons",
            "nameLocation": "middle",
            "nameGap": 20,
            "axisLine": {
                "lineStyle": {
                    "color": "#999"
                }
            },
            "splitLine": {
                "lineStyle": {
                    "type": "dashed"
                }
            }
        },
        "series": [
            {
                #"name": "2019",
                "data": data_2019_with_size,
                "type": "scatter",
                "emphasis": {
                    "focus": "series",
                    "itemStyle": {
                        "shadowBlur": 20,
                        "shadowColor": "rgba(120, 36, 50, 0.7)"
                    }
                },
                "itemStyle": {
                    "shadowBlur": 10,
                    "shadowColor": "rgba(120, 36, 50, 0.5)",
                    "shadowOffsetY": 5,
                    "color": {
                        "type": "radialGradient",
                        "x": 0.4,
                        "y": 0.3,
                        "r": 1,
                        "colorStops": [
                            {
                                "offset": 0,
                                "color": "rgb(251, 118, 123)"
                            },
                            {
                                "offset": 1,
                                "color": "rgb(204, 46, 72)"
                            }
                        ]
                    }
                }
            }
        ]
    }


    # Display the chart with adjustable height
    st_echarts(options=option)

@st.cache_data
def ideal_f():
    """
    Creates and displays an ideal donor profile tree chart.
    Using cache_data as the function returns deterministic output and doesn't involve widgets.
    """
    
    data = {
            "name": "Donneur idéal",
            "children": [
                {"name": "Statut professionnel", "children": [{"name": "Employé/ouvrier qualifié"}]},
                {"name": "Statut matrimonial", "children": [{"name": "Célibataire"}]},
                {"name": "Religion", "children": [{"name": "Christianisme"}]},
                {"name": "Âge", "children": [{"name": "26-35 ans"}]},
                {"name": "Niveau d'éducation", "children": [{"name": "Secondaire- Universitaire"}]},
                {"name": "Genre", "children": [{"name": "Homme"}]}
            ]
        }

    # Configuration des options pour le graphique (with dark pink color)
    option = {
        "tooltip": {
            "trigger": 'item',
            "triggerOn": 'mousemove'
        },
        "series": [
            {
                "type": 'tree',
                "data": [data],
                "top": '5%',
                "left": '20%',
                "bottom": '5%',
                "right": '30%',
                "symbolSize": 10,
                "label": {
                    "position": 'left',
                    "verticalAlign": 'middle',
                    "align": 'right',
                    "fontSize": 12,
                    "distance": 1
                },
                "leaves": {
                    "label": {
                        "position": 'right',
                        "verticalAlign": 'middle',
                        "align": 'left',
                        "fontSize": 12,
                        "distance": 0.7  # Reduced distance for leaves
                    }
                },
                "itemStyle": {  # Style for nodes
                    "color": "#C71585"  # Dark pink for nodes
                },
                "lineStyle": {  # Style for lines (edges)
                    "color": "#C71585",  # Dark pink for lines
                    "width": 1  # Optional: adjust line thickness
                },
                "emphasis": {
                    "focus": 'descendant',
                    "itemStyle": {
                        "color": "#FF69B4"  # Lighter pink on hover for contrast
                    },
                    "lineStyle": {
                        "color": "#FF69B4"
                    }
                },
                "expandAndCollapse": True,
                "animationDuration": 200,
                "animationDurationUpdate": 650
            }
        ]
    }

    return option

def display_ideal_chart_f():
    """
    Displays the ideal donor chart using the cached data.
    """
    option = ideal_f()
    st_echarts(options=option, height="200px")

@st.cache_data
def ideal_e():
    """
    Creates and displays an ideal donor profile tree chart.
    Using cache_data as the function returns deterministic output and doesn't involve widgets.
    """
    
    
    data = {
        "name": "Ideal Donor",
        "children": [
            {"name": "Professional Status", "children": [{"name": "Employee/Skilled Worker"}]},
            {"name": "Marital Status", "children": [{"name": "Single"}]},
            {"name": "Religion", "children": [{"name": "Christianity"}]},
            {"name": "Age", "children": [{"name": "26-35 years"}]},
            {"name": "Education Level", "children": [{"name": "Secondary - University"}]},
            {"name": "Gender", "children": [{"name": "Male"}]}
        ]
    }


    # Configuration des options pour le graphique (with dark pink color)
    option = {
        "tooltip": {
            "trigger": 'item',
            "triggerOn": 'mousemove'
        },
        "series": [
            {
                "type": 'tree',
                "data": [data],
                "top": '5%',
                "left": '20%',
                "bottom": '5%',
                "right": '30%',
                "symbolSize": 10,
                "label": {
                    "position": 'left',
                    "verticalAlign": 'middle',
                    "align": 'right',
                    "fontSize": 12,
                    "distance": 1
                },
                "leaves": {
                    "label": {
                        "position": 'right',
                        "verticalAlign": 'middle',
                        "align": 'left',
                        "fontSize": 12,
                        "distance": 0.7  # Reduced distance for leaves
                    }
                },
                "itemStyle": {  # Style for nodes
                    "color": "#C71585"  # Dark pink for nodes
                },
                "lineStyle": {  # Style for lines (edges)
                    "color": "#C71585",  # Dark pink for lines
                    "width": 1  # Optional: adjust line thickness
                },
                "emphasis": {
                    "focus": 'descendant',
                    "itemStyle": {
                        "color": "#FF69B4"  # Lighter pink on hover for contrast
                    },
                    "lineStyle": {
                        "color": "#FF69B4"
                    }
                },
                "expandAndCollapse": True,
                "animationDuration": 200,
                "animationDurationUpdate": 650
            }
        ]
    }

    return option

def display_ideal_chart_e():
    """
    Displays the ideal donor chart using the cached data.
    """
    option = ideal_e()
    st_echarts(options=option, height="200px")

def random_color():
    """Helper function to generate random hex color"""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


@st.cache_data
def prepare_douala_data(data):
    """
    Prepares and filters data for Douala.
    This function is cached as it performs data transformations that are deterministic for the same input.
    """
    # Remplacer les NaN par une chaîne vide pour éviter l'erreur
    data = data.copy()  # Create a copy to avoid modifying the original
    data['Arrondissement_de_résidence_'] = data['Arrondissement_de_résidence_'].fillna('')
    # Filtrer les données pour Douala uniquement
    douala_data = data[data['Arrondissement_de_résidence_'].str.contains('Douala')]
    return douala_data


@st.cache_data
def filter_by_demographic(_data, demographic_var, selected_value):
    """
    Filters data based on selected demographic variable and value.
    Cached as it's a data transformation that's deterministic for inputs.
    """
    if demographic_var != 'Aucune':
        return _data[_data[demographic_var] == selected_value]
    return _data


@st.cache_data
def calculate_eligibility_stats(filtered_data):
    """
    Calculates eligibility statistics by commune.
    Cached as it performs deterministic data processing.
    """
    # Obtenir les catégories uniques de ÉLIGIBILITÉ_AU_DON.
    categories = filtered_data['ÉLIGIBILITÉ_AU_DON.'].unique()

    # Regrouper par arrondissement et éligibilité pour calculer les proportions
    eligibility_by_commune = filtered_data.groupby(
        ['Arrondissement_de_résidence_', 'ÉLIGIBILITÉ_AU_DON.']
    ).size().unstack(fill_value=0)

    # Calculer le pourcentage pour chaque catégorie
    eligibility_by_commune['Total'] = eligibility_by_commune.sum(axis=1)
    for category in categories:
        eligibility_by_commune[f'{category}_%'] = (
            eligibility_by_commune.get(category, 0) / eligibility_by_commune['Total'] * 100
        ).round(2)

    return eligibility_by_commune, categories


@st.cache_data
def create_radar_options(eligibility_by_commune, categories):
    """
    Creates the radar chart options based on eligibility statistics.
    Cached as it generates chart options deterministically based on inputs.
    """
    communes = eligibility_by_commune.index.tolist()
    indicators = [{"name": commune, "max": 100} for commune in communes]

    # Créer les options pour chaque graphique radar
    radar_options = {}
    for category in categories:
        values = eligibility_by_commune[f'{category}_%'].tolist()
        # Définir la couleur : rouge pour "Eligible", aléatoire pour les autres
        line_color = "#FF0000" if category == "Eligible" else random_color()
        
        radar_options[category] = {
            "title": {
                "text": f"{category} (%)",
                "textStyle": {"fontSize": 12}
            },
            "tooltip": {},
            "radar": {
                "indicator": indicators,
                "shape": "circle",
                "splitNumber": 5,
                "splitLine": {  # Ajout pour définir la couleur des cercles concentriques
                    "lineStyle": {
                        "color": "gray",  # Noir pour les lignes des cercles
                        "width": 0.5
                    }
                },
                "axisName": {
                    "fontSize": 6,
                    "color": "#fff",
                    "backgroundColor": "#666",
                    "borderRadius": 2,
                    "padding": [2, 2.5]
                }
            },
            "series": [
                {
                    "name": category,
                    "type": "radar",
                    "data": [
                        {
                            "value": values,
                            "name": f"{category} (%)"
                        }
                    ],
                    "areaStyle": {"opacity": 0.2},
                    "lineStyle": {
                        "width": 1.5,
                        "color": line_color
                    }
                }
            ],
            "legend": {
                "data": [f"{category} (%)"],
                "top": "bottom",
                "textStyle": {"fontSize": 10}
            }
        }
    
    return radar_options


def three(data):
    """
    Main function to display radar charts for eligibility analysis.
    Not cached because it contains widgets and interactive elements.
    """
    # Prepare data
    douala_data = prepare_douala_data(data)
    
    # UI elements for filtering
    demographic_var = st.selectbox(
        "Choose a variable",
        options=['Aucune'] + [col for col in douala_data.columns if col not in ['Arrondissement_de_résidence_', 'ÉLIGIBILITÉ_AU_DON.']],
        index=0
    )
    
    # Filter based on selected demographic variable
    if demographic_var != 'Aucune':
        unique_values = douala_data[demographic_var].unique()
        selected_value = st.selectbox(f"Choisissez une valeur pour {demographic_var} :", options=unique_values)
        filtered_data = filter_by_demographic(douala_data, demographic_var, selected_value)
    else:
        filtered_data = douala_data
    
    # Calculate statistics
    eligibility_by_commune, categories = calculate_eligibility_stats(filtered_data)
    
    # Create radar chart options
    radar_options = create_radar_options(eligibility_by_commune, categories)
    
    # Display radar charts
    cols = st.columns([1, 1, 1])
    for i, category in enumerate(categories):
        with cols[i]:
            st_echarts(options=radar_options[category], height="300px", width="90%", key=f"radar_{category}")



@st.cache_resource
def heatmap(df) : 
    df['Date de remplissage de la fiche'] = pd.to_datetime(df['Date de remplissage de la fiche'], errors='coerce')

    # Filtrer les lignes où la date est valide
    df = df.dropna(subset=['Date de remplissage de la fiche'])

    # Compter le nombre de dons par jour
    dons_par_jour = df.groupby(df['Date de remplissage de la fiche'].dt.date).size().reset_index(name='Nombre de dons')

    # Générer une plage complète de dates pour inclure les jours sans dons
    date_min = dons_par_jour['Date de remplissage de la fiche'].min()
    date_max = dons_par_jour['Date de remplissage de la fiche'].max()
    all_dates = pd.date_range(start=date_min, end=date_max, freq='D')
    all_dates_df = pd.DataFrame({'Date': all_dates})
    all_dates_df['Date'] = all_dates_df['Date'].dt.date

    # Fusionner avec les données de dons pour inclure les jours à 0 dons
    dons_par_jour = all_dates_df.merge(dons_par_jour, left_on='Date', right_on='Date de remplissage de la fiche', how='left')
    dons_par_jour['Nombre de dons'] = dons_par_jour['Nombre de dons'].fillna(0)

    # Ajouter les colonnes nécessaires pour le heatmap
    dons_par_jour['Jour de la semaine'] = pd.to_datetime(dons_par_jour['Date']).dt.dayofweek
    dons_par_jour['Semaine'] = pd.to_datetime(dons_par_jour['Date']).dt.isocalendar().week
    dons_par_jour['Année'] = pd.to_datetime(dons_par_jour['Date']).dt.year

    # Définir les jours de la semaine (0 = Lundi, 6 = Dimanche)
    jours_semaine = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

    # Lister les années uniques
    annees_uniques = sorted(dons_par_jour['Année'].unique())

    # Palette de couleurs élégante : gris clair -> orange -> rouge profond
    custom_colorscale = [
        [0.0, 'rgb(240, 240, 240)'],  # Gris très clair
        [0.3, 'rgb(255, 204, 153)'],  # Orange doux
        [0.6, 'rgb(255, 102, 102)'],  # Rouge corail
        [1.0, 'rgb(153, 0, 0)']       # Rouge profond
    ]

    # Préparer les données pour chaque année
    traces = []
    for annee in annees_uniques:
        df_annee = dons_par_jour[dons_par_jour['Année'] == annee]
        semaines_uniques = sorted(df_annee['Semaine'].unique())
        
        # Créer une matrice pour le heatmap
        z_values = []
        for jour in range(7):
            row = []
            for semaine in semaines_uniques:
                data = df_annee[(df_annee['Jour de la semaine'] == jour) & (df_annee['Semaine'] == semaine)]
                row.append(data['Nombre de dons'].values[0] if not data.empty else 0)
            z_values.append(row)
        
        # Ajouter une trace pour chaque année avec visibilité initiale
        traces.append(
            go.Heatmap(
                z=z_values,
                x=semaines_uniques,
                y=jours_semaine,
                colorscale=custom_colorscale,
                showscale=True,
                colorbar=dict(
                    title='Nombre de dons',
                    tickfont=dict(size=12, color='black'),
                    thickness=20,
                    outlinecolor='black',
                    outlinewidth=1
                ),
                hoverinfo='x+y+z',
                zmin=0,
                hovertemplate='Semaine: %{x}<br>Jour: %{y}<br>Dons: %{z}<extra></extra>',
                visible=(annee == annees_uniques[0])  # Seule la première année visible au départ
            )
        )

    # Créer la figure avec toutes les traces
    fig = go.Figure(data=traces)

    # Ajouter le menu déroulant pour les années
    updatemenus = [
        dict(
            buttons=[
                dict(
                    args=[{
                        'visible': [annee == a for a in annees_uniques],
                        'title': f'Heatmap des dons - Année {annee}'
                    }],
                    label=str(annee),
                    method='update'
                ) for annee in annees_uniques
            ],
            direction='down',
            showactive=True,
            x=0.1,
            xanchor='left',
            y=1.15,
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='gray',
            font=dict(size=12, color='black')
        )
    ]

    # Personnaliser le layout pour un look moderne
    fig.update_layout(

        xaxis_title='Semaine de l\'année',
        yaxis_title='Jour de la semaine',
        
        
        updatemenus=updatemenus,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=80, r=40, t=100, b=40),
        showlegend=False,
        annotations=[
            dict(
                text="Source: vos données",
                xref="paper", yref="paper",
                x=1, y=-0.1,
                showarrow=False,
                font=dict(size=10, color='gray')
            )
        ]
    )

    # Ajouter des bordures légères aux cellules du heatmap
    fig.update_traces(
        zmin=0,
        connectgaps=False,
        xgap=1,  # Espacement horizontal entre les cellules
        ygap=1   # Espacement vertical entre les cellules
    )

    st.plotly_chart(fig,use_container_width=True)

def fiel_1(df) : 

    # Filtrer pour ne garder que les donneurs ayant répondu "Oui" à "A-t-il (elle) déjà donné le sang"
    df = df[df['A-t-il (elle) déjà donné le sang'] == 'Oui']

    # Étape 2 : Créer les flux entre chaque paire de catégories consécutives
    # On va créer des paires : (A-t-il (elle) déjà donné le sang → ÉLIGIBILITÉ_AU_DON.), (ÉLIGIBILITÉ_AU_DON. → Classe_Age), etc.

    # Flux 1 : A-t-il (elle) déjà donné le sang → ÉLIGIBILITÉ_AU_DON.
    flux1 = df.groupby(['A-t-il (elle) déjà donné le sang', 'ÉLIGIBILITÉ_AU_DON.']).size().reset_index(name='Nombre')

    # Flux 2 : ÉLIGIBILITÉ_AU_DON. → Classe_Age
    flux2 = df.groupby(['ÉLIGIBILITÉ_AU_DON.', 'Classe_Age']).size().reset_index(name='Nombre')

    # Flux 3 : Classe_Age → Genre_
    flux3 = df.groupby(['Classe_Age', 'Genre_']).size().reset_index(name='Nombre')

    # Flux 4 : Genre_ → Niveau_d_etude
    flux4 = df.groupby(['Genre_', "Niveau_d'etude"]).size().reset_index(name='Nombre')

    # Flux 5 : Niveau_d_etude → Religion_Catégorie
    flux5 = df.groupby(["Niveau_d'etude", 'Religion_Catégorie']).size().reset_index(name='Nombre')

    # Flux 6 : Religion_Catégorie → Situation_Matrimoniale_(SM)
    flux6 = df.groupby(['Religion_Catégorie', 'Situation_Matrimoniale_(SM)']).size().reset_index(name='Nombre')

    # Flux 7 : Situation_Matrimoniale_(SM) → categories
    flux7 = df.groupby(['Situation_Matrimoniale_(SM)', 'categories']).size().reset_index(name='Nombre')

    # Étape 3 : Créer la liste des nœuds (toutes les catégories uniques)
    donne_sang = df['A-t-il (elle) déjà donné le sang'].unique().tolist()  # Contient uniquement "Oui" après le filtre
    eligibilites = df['ÉLIGIBILITÉ_AU_DON.'].unique().tolist()
    classes_age = df['Classe_Age'].unique().tolist()
    genres = df['Genre_'].unique().tolist()
    niveaux_etude = df["Niveau_d'etude"].unique().tolist()
    religions = df['Religion_Catégorie'].unique().tolist()
    situations_matrimoniales = df['Situation_Matrimoniale_(SM)'].unique().tolist()
    categories = df['categories'].unique().tolist()

    # Liste complète des nœuds
    nodes = (donne_sang + eligibilites + classes_age + genres + niveaux_etude +
        religions + situations_matrimoniales + categories)

    # Étape 4 : Créer un dictionnaire pour mapper les nœuds à des indices
    node_dict = {node: idx for idx, node in enumerate(nodes)}

    # Étape 5 : Créer les liens (source, target, value) pour chaque flux
    # Liens pour Flux 1 : A-t-il (elle) déjà donné le sang → ÉLIGIBILITÉ_AU_DON.
    source1 = flux1['A-t-il (elle) déjà donné le sang'].map(node_dict).tolist()
    target1 = flux1['ÉLIGIBILITÉ_AU_DON.'].map(node_dict).tolist()
    value1 = flux1['Nombre'].tolist()

    # Liens pour Flux 2 : ÉLIGIBILITÉ_AU_DON. → Classe_Age
    source2 = flux2['ÉLIGIBILITÉ_AU_DON.'].map(node_dict).tolist()
    target2 = flux2['Classe_Age'].map(node_dict).tolist()
    value2 = flux2['Nombre'].tolist()

    # Liens pour Flux 3 : Classe_Age → Genre_
    source3 = flux3['Classe_Age'].map(node_dict).tolist()
    target3 = flux3['Genre_'].map(node_dict).tolist()
    value3 = flux3['Nombre'].tolist()

    # Liens pour Flux 4 : Genre_ → Niveau_d_etude
    source4 = flux4['Genre_'].map(node_dict).tolist()
    target4 = flux4["Niveau_d'etude"].map(node_dict).tolist()
    value4 = flux4['Nombre'].tolist()

    # Liens pour Flux 5 : Niveau_d_etude → Religion_Catégorie
    source5 = flux5["Niveau_d'etude"].map(node_dict).tolist()
    target5 = flux5['Religion_Catégorie'].map(node_dict).tolist()
    value5 = flux5['Nombre'].tolist()

    # Liens pour Flux 6 : Religion_Catégorie → Situation_Matrimoniale_(SM)
    source6 = flux6['Religion_Catégorie'].map(node_dict).tolist()
    target6 = flux6['Situation_Matrimoniale_(SM)'].map(node_dict).tolist()
    value6 = flux6['Nombre'].tolist()

    # Liens pour Flux 7 : Situation_Matrimoniale_(SM) → categories
    source7 = flux7['Situation_Matrimoniale_(SM)'].map(node_dict).tolist()
    target7 = flux7['categories'].map(node_dict).tolist()
    value7 = flux7['Nombre'].tolist()

    # Combiner tous les liens
    source = source1 + source2 + source3 + source4 + source5 + source6 + source7
    target = target1 + target2 + target3 + target4 + target5 + target6 + target7
    value = value1 + value2 + value3 + value4 + value5 + value6 + value7

    # Étape 6 : Définir les couleurs pour les nœuds
    # On attribue des couleurs différentes pour chaque groupe de nœuds
    num_donne_sang = len(donne_sang)
    num_eligibilites = len(eligibilites)
    num_classes_age = len(classes_age)
    num_genres = len(genres)
    num_niveaux_etude = len(niveaux_etude)
    num_religions = len(religions)
    num_situations_matrimoniales = len(situations_matrimoniales)
    num_categories = len(categories)

    # Couleurs pour chaque groupe
    colors_donne_sang = ['#ff00ff'] * num_donne_sang  # Magenta pour A-t-il (elle) déjà donné le sang
    colors_eligibilites = ['#00cc96'] * num_eligibilites  # Vert pour ÉLIGIBILITÉ_AU_DON.
    colors_classes_age = ['#f83e8c'] * num_classes_age  # Rose pour Classe_Age
    colors_genres = ['#8b008b'] * num_genres  # Violet foncé pour Genre_
    colors_niveaux_etude = ['#1e90ff'] * num_niveaux_etude  # Bleu pour Niveau_d_etude
    colors_religions = ['#ffd700'] * num_religions  # Jaune pour Religion_Catégorie
    colors_situations_matrimoniales = ['#ff4500'] * num_situations_matrimoniales  # Orange pour Situation_Matrimoniale_(SM)
    colors_categories = ['#00b7eb'] * num_categories  # Cyan pour categories

    # Combiner les couleurs
    node_colors = (colors_donne_sang + colors_eligibilites + colors_classes_age + colors_genres +
            colors_niveaux_etude + colors_religions + colors_situations_matrimoniales + colors_categories)

    # Étape 7 : Créer le diagramme de Sankey
    fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=nodes,
        color=node_colors
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color='rgba(200, 200, 200, 0.5)'  # Couleur des liens
    )
    )])

    # Étape 8 : Personnaliser le layout
    fig.update_layout(
    title_text="Flux des donneurs (A-t-il donné = Oui) : A-t-il donné → Éligibilité → Classe d'âge → Genre → Niveau d'étude → Religion → Situation matrimoniale → Catégorie",
    font=dict(size=10, color='black'),
    width=1200,  # Ajuster la largeur pour une meilleure lisibilité
    height=800  # Ajuster la hauteur
    )
    st.plotly_chart(fig,use_container_width=True)

def plot_top4_demographic(data_recurrent, data_non_recurrent, column, title_prefix, comparison=False, orientation='v'):
    """
    Génère un graphique Plotly avec les top 4 catégories pour une variable démographique.
    
    Parameters:
    - data_recurrent: DataFrame des donneurs récurrents
    - data_non_recurrent: DataFrame des donneurs non récurrents (pour comparaison)
    - column: Colonne démographique à analyser
    - title_prefix: Préfixe du titre du graphique
    - comparison: Booléen pour indiquer si on compare récurrents et non récurrents
    - orientation: 'v' pour vertical (par défaut), 'h' pour horizontal
    """
    # Compter les occurrences pour les donneurs récurrents et non récurrents
    if comparison:
        # Pour les graphiques de comparaison (récurrents vs non récurrents)
        count_recurrent = data_recurrent[column].value_counts()
        count_non_recurrent = data_non_recurrent[column].value_counts()
        
        # Fusionner les deux séries pour obtenir toutes les catégories
        all_categories = pd.concat([count_recurrent, count_non_recurrent], axis=1, sort=False)
        all_categories.columns = ['Récurrents', 'Non Récurrents']
        all_categories.fillna(0, inplace=True)
        
        # Calculer le total pour trier
        all_categories['Total'] = all_categories['Récurrents'] + all_categories['Non Récurrents']
        top4_categories = all_categories.sort_values('Total', ascending=False).head(4).index
        
        # Filtrer les données pour ne garder que les top 4 catégories
        count_recurrent = count_recurrent[count_recurrent.index.isin(top4_categories)]
        count_non_recurrent = count_non_recurrent[count_non_recurrent.index.isin(top4_categories)]
        
        # Créer le graphique de comparaison
        fig = go.Figure()
        
        if orientation == 'v':
            # Orientation verticale
            fig.add_trace(go.Bar(
                x=count_recurrent.index,
                y=count_recurrent.values,
                name='Récurrents (Oui)',
                marker_color='#00cc96',  # Vert
                text=count_recurrent.values,
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Nombre: %{y}<br>Catégorie: Récurrents<extra></extra>'
            ))
            
            fig.add_trace(go.Bar(
                x=count_non_recurrent.index,
                y=count_non_recurrent.values,
                name='Non Récurrents (Non)',
                marker_color='#ff5733',  # Orange
                text=count_non_recurrent.values,
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Nombre: %{y}<br>Catégorie: Non Récurrents<extra></extra>'
            ))
            
            xaxis_title = column
            yaxis_title = 'Nombre de donneurs'
            xaxis_config = dict(tickangle=45, title_standoff=25)
            yaxis_config = dict(gridcolor='rgba(0,0,0,0.1)', title_standoff=25)
            
        else:
            # Orientation horizontale
            fig.add_trace(go.Bar(
                y=count_recurrent.index,
                x=count_recurrent.values,
                name='Récurrents (Oui)',
                marker_color='#00cc96',  # Vert
                text=count_recurrent.values,
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Nombre: %{x}<br>Catégorie: Récurrents<extra></extra>',
                orientation='h'
            ))
            
            fig.add_trace(go.Bar(
                y=count_non_recurrent.index,
                x=count_non_recurrent.values,
                name='Non Récurrents (Non)',
                marker_color='#ff5733',  # Orange
                text=count_non_recurrent.values,
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Nombre: %{x}<br>Catégorie: Non Récurrents<extra></extra>',
                orientation='h'
            ))
            
            xaxis_title = 'Nombre de donneurs'
            yaxis_title = column
            xaxis_config = dict(gridcolor='rgba(0,0,0,0.1)', title_standoff=25)
            yaxis_config = dict(title_standoff=25)
            
    else:
        # Pour les graphiques de distribution (donneurs récurrents uniquement)
        count_recurrent = data_recurrent[column].value_counts()
        top4_categories = count_recurrent.head(4).index
        count_recurrent = count_recurrent[count_recurrent.index.isin(top4_categories)]
        
        # Créer le graphique de distribution
        fig = go.Figure()
        
        if orientation == 'v':
            # Orientation verticale
            fig.add_trace(go.Bar(
                x=count_recurrent.index,
                y=count_recurrent.values,
                name='Récurrents',
                marker_color='#00cc96',  # Vert
                text=count_recurrent.values,
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Nombre: %{y}<br>Catégorie: Récurrents<extra></extra>'
            ))
            
            xaxis_title = column
            yaxis_title = 'Nombre de donneurs'
            xaxis_config = dict(tickangle=45, title_standoff=25)
            yaxis_config = dict(gridcolor='rgba(0,0,0,0.1)', title_standoff=25)
            
        else:
            # Orientation horizontale
            fig.add_trace(go.Bar(
                y=count_recurrent.index,
                x=count_recurrent.values,
                name='Récurrents',
                marker_color='#00cc96',  # Vert
                text=count_recurrent.values,
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Nombre: %{x}<br>Catégorie: Récurrents<extra></extra>',
                orientation='h'
            ))
            
            xaxis_title = 'Nombre de donneurs'
            yaxis_title = column
            xaxis_config = dict(gridcolor='rgba(0,0,0,0.1)', title_standoff=25)
            yaxis_config = dict(title_standoff=25)
    
    # Personnaliser le layout
    fig.update_layout(
        title=f"{title_prefix} par {column} (Top 4)",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        legend=dict(
            title='Statut de récurrence',
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12, color='black'),
        margin=dict(l=50, r=50, t=80, b=150),
        width=800,
        height=600,
        bargap=0.2,
        barmode='group' if comparison else 'stack'
    )
    
    # Afficher le graphique
    st.plotly_chart(fig)
    orientation_dict = {
'Classe_Age': 'v',  # Vertical pour les tranches d'âge
'Genre_': 'v',      # Vertical pour le genre
'Niveau_d_etude': 'v',  # Vertical pour le niveau d'étude
'Religion_Catégorie': 'h',  # Horizontal pour les religions (étiquettes longues)
'Situation_Matrimoniale_(SM)': 'v',  # Vertical pour la situation matrimoniale
'categories': 'h',  # Horizontal pour les catégories professionnelles (étiquettes longues)
'Arrondissement_de_résidence_': 'h'  # Horizontal pour les arrondissements (étiquettes longues)
}

