import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import matplotlib.patheffects as PathEffects
import streamlit as st
import numpy as np
import json

dir_detectors   = "data/detectors/"
dir_sources     = "data/sources/"
label_linewidth = 2
label_color     = "#ffffff"

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def gradient_fill(x, y, fill_color=None, ymin=0, ax=None, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.

    FROM: https://stackoverflow.com/questions/29321835/is-it-possible-to-get-color-gradients-under-curve-in-matplotlib
    """
    if ax is None:
        ax = plt.gca()

    alpha= .8
    z = np.empty((100, 1, 4), dtype=float)

    rgb = mcolors.to_rgb(fill_color)
    z[:,:,:3] = rgb
    z[:,:,-1] = np.linspace(0, alpha, 100)[:,None]

    # xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    xmin, xmax, ymin, ymax = x.min(), x.max(), ymin, y.max()


    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower')

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)
    return im



def data2plot(data, plottype):
    if plottype == 1:
        x = np.log10(data[:,0])
        y = np.log10(data[:,1])

    elif plottype == 0:
        H0 = 3.240779291010696e-18
        x = np.log10(data[:,0])
        y = np.log10(2*(np.pi*data[:,1]*x/H0)**2)
    return x, y 


# function toEnergySpec(data) {
# 	var H0 = 3.240779291010696e-18;
# 	return data.map(function(val) {
# 		var f = val[0];
# 		var hc = val[1];
# 		return [f, 2*Math.pow(Math.PI*f*hc/H0,2)];
# 	});
# }



st.set_page_config(page_title="GW Plotter", page_icon=":dizzy:", layout="wide")
st.title("GW Plotter 1.1 beta")

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_DET():
    def_curve_col = "#000000"

    NDET         = 19
    dict_DET     = [[]]*NDET
    dict_DET[0]  = {'todo':0,'color':def_curve_col,'label':'EPTA','label_x':2e-6,'label_y':5e-13  , 'N_pulsars':5,  'ObsTime':10,'ObsRate':14,'TimingPrec':1e-7}
    dict_DET[1]  = {'todo':1,'color':def_curve_col,'label':'IPTA','label_x':2e-6,'label_y':1e-13  , 'N_pulsars':20, 'ObsTime':15,'ObsRate':14,'TimingPrec':1e-7}
    dict_DET[2]  = {'todo':0,'color':def_curve_col,'label':'SKA' ,'label_x':2e-6,'label_y':7.5e-15, 'N_pulsars':100,'ObsTime':20,'ObsRate':14,'TimingPrec':3e-8}

    dict_DET[3]  = {'todo':0,'color':def_curve_col,'label':'eLISA'   ,'label_x':1.5e-6,'label_y':7e-17}
    dict_DET[4]  = {'todo':1,'color':def_curve_col,'label':'LISA'    ,'label_x':2e-6  ,'label_y':1e-18}
    dict_DET[5]  = {'todo':0,'color':def_curve_col,'label':'DECIGO'  ,'label_x':7e-2  ,'label_y':1e-23}
    dict_DET[6]  = {'todo':0,'color':def_curve_col,'label':'BBO'     ,'label_x':4.2e-3,'label_y':2e-24}
    dict_DET[7]  = {'todo':0,'color':def_curve_col,'label':'ALIA'    ,'label_x':1e-1  ,'label_y':7e-22}
    dict_DET[8]  = {'todo':0,'color':def_curve_col,'label':'TianQin' ,'label_x':2.5e+0,'label_y':1.5e-18}

    dict_DET[9]  = {'todo':0,'color':def_curve_col,'label':'GEO'           ,'label_x':5e+4  ,'label_y':1.55e-18}
    dict_DET[10] = {'todo':0,'color':def_curve_col,'label':'LIGO'          ,'label_x':5e+4  ,'label_y':7.99e-19}
    dict_DET[11] = {'todo':0,'color':def_curve_col,'label':'aLIGO-O1'      ,'label_x':5e+4  ,'label_y':1.1e-19}
    dict_DET[12] = {'todo':1,'color':def_curve_col,'label':'aLIGOD'        ,'label_x':5e+4  ,'label_y':5.68e-20}
    dict_DET[13] = {'todo':0,'color':def_curve_col,'label':'ApLIGO'        ,'label_x':5e+4  ,'label_y':1.51e-20}
    dict_DET[14] = {'todo':0,'color':def_curve_col,'label':'Virgo'         ,'label_x':5e+4  ,'label_y':4.12e-19}
    dict_DET[15] = {'todo':0,'color':def_curve_col,'label':'aVirgo'        ,'label_x':5e+4  ,'label_y':2.13e-19}
    dict_DET[16] = {'todo':0,'color':def_curve_col,'label':'KAGRA'         ,'label_x':5e+4  ,'label_y':2.92e-20}
    dict_DET[17] = {'todo':0,'color':def_curve_col,'label':'ET'            ,'label_x':5e+4  ,'label_y':7.55e-21}
    dict_DET[18] = {'todo':0,'color':def_curve_col,'label':'CE'            ,'label_x':5e+4  ,'label_y':3.9e-21}

    # Load sensitivity data
    for i in range(NDET):
        with open(dir_detectors+dict_DET[i]["label"]+'.json', "r") as read_file:
            data = json.load(read_file)
            dict_DET[i]["data"] = data["data"]

    return dict_DET



@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_SOURCES():

    NSOURCES        = 11
    dict_SOURCES    = [[]]*NSOURCES
    dict_SOURCES[0] = {'todo':1,'color':'#cc6600','label':'Stochastic\nbackground'         ,'label_x':7.6e-11 ,'label_y':3.7e-14}
    dict_SOURCES[1] = {'todo':0,'color':'#e69900','label':'Supermassive\nbinaries'         ,'label_x':3.5e-7  ,'label_y':1e-17}
    dict_SOURCES[2] = {'todo':0,'color':'#59b2e6','label':'Resolvable\ngalactic binaries'  ,'label_x':2.5e-3  ,'label_y':1.5e-18}
    dict_SOURCES[3] = {'todo':0,'color':'#00ff00','label':'Unresolvable\ngalactic binaries','label_x':5e-7    ,'label_y':1e-21}
    dict_SOURCES[4] = {'todo':1,'color':'#0073b2','label':'Massive\nbinaries'              ,'label_x':1e-4    ,'label_y':2e-17}
    dict_SOURCES[5] = {'todo':1,'color':'#009980','label':'Extreme mass\nratio inspirals'  ,'label_x':9e-7    ,'label_y':2e-20}
    dict_SOURCES[6] = {'todo':1,'color':'#a0a1c8','label':'Type IA\nsupernovae'           ,'label_x':3.3e-1  ,'label_y':2e-20}
    dict_SOURCES[7] = {'todo':1,'color':'#cc99b2','label':'Compact binary\ninspirals'      ,'label_x':3e+3    ,'label_y':3e-23}
    dict_SOURCES[8] = {'todo':0,'color':'#800080','label':'Core collapse\nsupernovae'      ,'label_x':3e+3    ,'label_y':4e-24, 'Dist':0.3}
    dict_SOURCES[9] = {'todo':0,'color':'#ffffff','label':'Pulsars'                        ,'label_x':8e+4    ,'label_y':5e-25, 'AmplScale':1}
    dict_SOURCES[10]= {'todo':1,'color':'#ff0000','label':'GW150914'                       ,'label_x':3e+1    ,'label_y':7e-21}

    alias_names    = ["BKG", "SMBBH", "GalBinRes", "GalBinUnres",
                    "MBBH", "EMRI", "SNIa", "CBC", "CCSN", "PSR", "GW150914" ]

    for i in range(NSOURCES):
        with open(dir_sources+alias_names[i]+'.json', "r") as read_file:
            data = json.load(read_file)
            dict_SOURCES[i]["data"] = data["data"]

    return dict_SOURCES, alias_names


def ff_alias_SOURCES(name):
    ind = all_SOURCES.index(name)
    return alias_SOURCES[ind]



st.sidebar.header("🪄 Settings")

# Select plot type
plottypes = ["Characteristic Strain", "Power Spectral Density"]
def headerlabel(number):
    return "{1}".format(number, plottypes[number-1])
plottype = st.sidebar.radio('Select plot type:', [1,0], format_func=headerlabel)

# Load (only for the first time) the baseline settings
dict_DET  = load_DET()
NDET      = len(dict_DET)
all_DET   = [dict_DET[i]["label"] for i in range(NDET)]
todo_DET  = [dict_DET[i]['label'] for i in range(NDET) if dict_DET[i]['todo']==1]

dict_SOURCES, alias_SOURCES = load_SOURCES()
NSOURCES                    = len(dict_SOURCES)
all_SOURCES                 = [dict_SOURCES[i]["label"] for i in range(NSOURCES)]
todo_SOURCES                = [dict_SOURCES[i]['label'] for i in range(NSOURCES) if dict_SOURCES[i]['todo']==1]


# Multiselect entries
todo_DET         = st.sidebar.multiselect('Detectors', all_DET , default=todo_DET)
todo_SOURCES     = st.sidebar.multiselect('Sources', all_SOURCES, default=todo_SOURCES, format_func=ff_alias_SOURCES)
ind_todo_DET     = [all_DET.index(i) for i in todo_DET]
ind_todo_SOURCES = [all_SOURCES.index(i) for i in todo_SOURCES]


# Plot Settings
st.sidebar.write("Elements")

xlims = np.array([0.4e-10, 1.4e+6])
ylims = np.array([1e-26, 1.1e-12])
col1, col2= st.sidebar.columns(2)

with col1:
    xlims[0] = st.number_input(label="$x_{min}$", format="%.1e", value=xlims[0], step=xlims[0]/10)
    ylims[0] = st.number_input(label="$y_{min}$", format="%.1e", value=ylims[0], step=ylims[0]/10)
with col2:
    xlims[1] = st.number_input(label="$x_{max}$", format="%.1e", value=xlims[1], step=xlims[1]/10)
    ylims[1] = st.number_input(label="$y_{max}$", format="%.1e", value=ylims[1], step=ylims[1]/10)

col1, col2 = st.sidebar.columns(2)
with col1:
    label_linewidth = col1.number_input("Label contur width", 0, 10, label_linewidth)
with col2:
    label_color = col2.color_picker("Label color", label_color) 

todo_theme = st.sidebar.selectbox("Theme", ["Light", "Dark"])


if todo_theme == "Dark":
    plt.style.use("dark_background")
    label_linewidth = 0
    for i in range(NDET):
        dict_DET[i]['color'] = "#ffffff"      

if todo_theme == "Light":
    label_linewidth = 2
    label_color     = "#ffffff"
    plt.style.use("default")
    for i in range(NDET):
        dict_DET[i]['color'] = "#000000"  




        


# Advanced settings
st.sidebar.subheader("Advanced settings")
todo_type  = st.sidebar.selectbox("", ["Detector", "Source"])
# st.markdown(
#     """
#     <style>
#     [data-baseweb="select"] {
#         margin-top: -40px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
if todo_type == "Detector":
    adv_DET    = st.sidebar.selectbox("Type", all_DET)
    ind        = all_DET.index(adv_DET)
    col1, col2 = st.sidebar.columns(2)

    with col1:
        dict_DET[ind]['label_x'] = st.number_input(label="$x_{label}$", format="%.1e", 
                                                value=dict_DET[ind]['label_x'], 
                                                step=dict_DET[ind]['label_x']/2)
        dict_DET[ind]['label']   = st.text_input('Label',dict_DET[ind]['label'])
    with col2:
        dict_DET[ind]['label_y'] = st.number_input(label="$y_{label}$", format="%.1e", 
                                                value=dict_DET[ind]['label_y'], 
                                                step=dict_DET[ind]['label_y']/2)
        dict_DET[ind]['color']   = st.color_picker("Color", value=dict_DET[ind]['color'])

elif todo_type == "Source":
    adv_SOURCE = st.sidebar.selectbox("Type", all_SOURCES)
    ind        = all_SOURCES.index(adv_SOURCE)
    col1, col2 = st.sidebar.columns(2)

    with col1:
        dict_SOURCES[ind]['label_x'] = st.number_input(label="$x_{label}$", format="%.1e", 
                                                value=dict_SOURCES[ind]['label_x'], 
                                                step=dict_SOURCES[ind]['label_x']/2)
        dict_SOURCES[ind]['label']   = st.text_input('Label',dict_SOURCES[ind]['label'])
    with col2:
        dict_SOURCES[ind]['label_y'] = st.number_input(label="$y_{label}$", format="%.1e", 
                                                value=dict_SOURCES[ind]['label_y'], 
                                                step=dict_SOURCES[ind]['label_y']/2)
        dict_SOURCES[ind]['color']   = st.color_picker("Color", value=dict_SOURCES[ind]['color'])







if st.sidebar.button('Reset'):
    st.runtime.legacy_caching.clear_cache()
    st.runtime.legacy_caching.clear_cache()





#################################################### PLOTTING

fig, ax = plt.subplots(figsize=(9,5),dpi=300)
for ind in ind_todo_DET:
    # Plot Detectors
    tdet    = dict_DET[ind]
    x, y = data2plot(np.array(tdet["data"]), plottype)

    ax.plot(x, y, color=tdet["color"])
    txt = ax.text(s=tdet["label"], x=np.log10(tdet["label_x"]), y=np.log10(tdet["label_y"]), 
                  fontsize=8, fontweight='bold', color=tdet["color"])
    txt.set_path_effects([PathEffects.withStroke(linewidth=label_linewidth, foreground=label_color)])

for ind in ind_todo_SOURCES:
    # Plot Sources
    tsource = dict_SOURCES[ind]
    x, y = data2plot(np.array(tsource["data"]), plottype)

    gradient_fill(x,y,fill_color=tsource["color"],c=tsource["color"],ymin=np.log10(ylims[0]),ax=ax)
    txt = ax.text(s=tsource["label"], x=np.log10(tsource["label_x"]), y=np.log10(tsource["label_y"]),
                  fontsize=8, ha="left", fontweight='bold',color=tsource["color"])

    txt.set_path_effects([PathEffects.withStroke(linewidth=label_linewidth, foreground=label_color)])

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Characteristic Strain")
ax.set_xlim(*np.log10(xlims))
ax.set_ylim(*np.log10(ylims))

ax.xaxis.set_major_formatter('10$^{{{x:.0f}}}$')
ax.yaxis.set_major_formatter('10$^{{{x:.0f}}}$')
# ax.set_xscale('log')
# ax.set_yscale('log')


fn = "gwplotter.png"
plt.tight_layout()
plt.savefig(fn, transparent=True)
with open(fn, "rb") as img:
    btn = st.download_button(
        label="Download image",
        data=img,
        file_name=fn,
        mime="image/png"
    )
st.pyplot(fig)

st.write("")


########################################## REFS
st.write("")
st.subheader("References")
st.markdown("[http://gwplotter.com/](http://gwplotter.com/)")
st.write("TBD")



























########################################################################################## JUNK

# st.write(dict_DET[ind]['label'])

#     # for i in range(NDET):
    #     st.write(dict_DET[i]["label"])


# d = [dict_PT, dict_SPACE, dict_GROUND]
# d = [*dict_PT, *dict_SPACE , *dict_GROUND]
# t = [todo_PT, todo_SPACE, todo_GROUND]
# n = ["*Pulsar Timing*", "Space-based", "Ground-based"]




# cols = st.columns(len(d[k]))
# for i, c in enumerate(cols):
#     with c:
#         t[k][i] = st.checkbox(d[k][i]['label'])


# for k in range(3):
#     st.write(n[k])
#     cols = st.columns(len(d[k]))
#     for i, c in enumerate(cols):
#         with c:
#             t[k][i] = st.checkbox(d[k][i]['label'])



# st.subheader("Sources")

# cols = st.columns(len(dict_PT))
# for i, c in enumerate(cols):
#     with c:
#         todo_PT[i] = st.checkbox(dict_PT[i]['label'])


# todo_PT[0] = st.checkbox('EPTA')
# todo_PT[1] = st.checkbox('IPTA')
# todo_PT[1] = st.checkbox('SKA')



# todo_PT = [0]*3
# dict_PT = [[]]*3
# dict_PT[0] = {'N_pulsars':5,  'ObsTime':10,'ObsRate':14,'TimingPrec':1e-7,'color':def_curve_col,'label':'EPTA','label_x':2e-6,'label_y':5e-13}
# dict_PT[1] = {'N_pulsars':20, 'ObsTime':15,'ObsRate':14,'TimingPrec':1e-7,'color':def_curve_col,'label':'IPTA','label_x':2e-6,'label_y':1e-13}
# dict_PT[2] = {'N_pulsars':100,'ObsTime':20,'ObsRate':14,'TimingPrec':3e-8,'color':def_curve_col,'label':'SKA' ,'label_x':2e-6,'label_y':7.5e-15}

# todo_SPACE = [0]*6
# dict_SPACE = [[]]*6
# dict_SPACE[0] = {'color':def_curve_col,'label':'eLISA'   ,'label_x':1.5e-6,'label_y':7e-17}
# dict_SPACE[1] = {'color':def_curve_col,'label':'LISA'    ,'label_x':2e-6  ,'label_y':1e-18}
# dict_SPACE[2] = {'color':def_curve_col,'label':'DECIGO'  ,'label_x':7e-2  ,'label_y':1e-23}
# dict_SPACE[3] = {'color':def_curve_col,'label':'BBO'     ,'label_x':4.2e-3,'label_y':2e-24}
# dict_SPACE[4] = {'color':def_curve_col,'label':'ALIA'    ,'label_x':1e-1  ,'label_y':7e-22}
# dict_SPACE[5] = {'color':def_curve_col,'label':'TianQuin','label_x':2.5e+0,'label_y':1.5e-18}

# todo_GROUND = [0]*10
# dict_GROUND = [[]]*10
# dict_GROUND[0] = {'color':def_curve_col,'label':'GEO'           ,'label_x':8e+4  ,'label_y':1.55e-18}
# dict_GROUND[1] = {'color':def_curve_col,'label':'LIGO'          ,'label_x':8e+4  ,'label_y':7.99e-19}
# dict_GROUND[2] = {'color':def_curve_col,'label':'aLIGO (O1)'    ,'label_x':4e+4  ,'label_y':1.1e-19}
# dict_GROUND[3] = {'color':def_curve_col,'label':'aLIGO (des)'   ,'label_x':8e+4  ,'label_y':5.68e-20}
# dict_GROUND[4] = {'color':def_curve_col,'label':'LIGO A+'       ,'label_x':8e+4  ,'label_y':1.51e-20}
# dict_GROUND[5] = {'color':def_curve_col,'label':'Virgo'         ,'label_x':8e+4  ,'label_y':4.12e-19}
# dict_GROUND[6] = {'color':def_curve_col,'label':'Virgo Adv'     ,'label_x':8e+4  ,'label_y':2.13e-19}
# dict_GROUND[7] = {'color':def_curve_col,'label':'KAGRA'         ,'label_x':8e+4  ,'label_y':2.92e-20}
# dict_GROUND[8] = {'color':def_curve_col,'label':'ET'            ,'label_x':8e+4  ,'label_y':7.55e-21}
# dict_GROUND[9] = {'color':def_curve_col,'label':'CE'            ,'label_x':8e+4  ,'label_y':3.9e-21}