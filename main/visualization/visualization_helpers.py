from datetime import datetime
from algorithm.planner import JourneyPlanner
from collections import defaultdict
import pickle
import ipywidgets as widgets
from IPython.display import display, clear_output
import datetime
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import plotly
import matplotlib.pyplot as plt
import pickle
from algorithm.journey import *

def visualize_map_scattermapbox(journey, routes, plot_number = 0):
    color_dict = {}
    route_types = np.unique(routes.routes_desc)
    colors = ["red", "yellow",
            "brown",
           "black",
            "cyan",
            "navy",
            "linen",
            "white",
            "tomato",
            "green",
           "purple",
            "olive",
            "lime",
            "orange",
            "greenyellow"]

    for x in zip(route_types, colors):
        color_dict[x[0]] = x[1]
    
    color_dict["Walking"] = "blue" 
    
    
    # l is the layout of the plot, we use to set the title, the margins and the style of the map
    departure_station = journey.links[0].dep_stop['stop_name']
    arrival_station = journey.links[-1].arr_stop['stop_name']
    l = go.Layout(
        title= f'Journey Planner from {departure_station} to {arrival_station}: suggested trip n. {plot_number}, <br> confidence:{journey.confidence}, number of connections: {journey.num_connections}, total expected walking time: {journey.total_duration_footpaths} sec', 
        margin ={'l':0,'t':100,'b':0,'r':0},
        mapbox = {
            'center': {'lon': 10, 'lat': 10},
            'style': "open-street-map",
            'center': {'lon':8.57 , 'lat': 47.40},
            'zoom': 9.8,},
        title_x = 0.41
    )
    
    # we initialize the map with the layout we created above
    
    fig = go.Figure(go.Scattermapbox(
            mode = "markers+lines",
            marker = {'size': 10},
            ), l)
    
    # here we iterate through the path in order to plot each part of the trip
    i = 0
    for x in journey.links:
        
        if isinstance(x, Change):  # skipping changes, Changes should not be plotted
            continue
            
        #we should not plot walks where you stay inside the same station
        if isinstance(x, Footpath):
            if x.dep_stop['parent_station'] == x.arr_stop['parent_station'] and x.dep_stop['parent_station'] != '': 
                continue
            transport_type = "Walking" # this is used later for plotting the legend
            
        #getting the transport type of the trip, used for plotting the legend
        if isinstance(x, Trip):
            transport_type = routes.drop_duplicates("route_id").set_index("route_id").loc[x.trip['route_id']]["routes_desc"] 
            

        
        arr_df = pd.DataFrame(x.arr_stop, index = ["arr"])
        dep_df = pd.DataFrame(x.dep_stop, index = ["dep"])
        #to plot the lines we merge the arrival and departure stop in one df
        arr_dep_df = pd.concat([arr_df, dep_df])
        
        dep_time_date = datetime.fromtimestamp(x.dep_time)
        arr_time_date = datetime.fromtimestamp(x.arr_time) if isinstance(x, Trip) else datetime.fromtimestamp(x.dep_time + x.duration)
        hover_string = f'station name: {x.dep_stop["stop_name"]} <br>departure time: {dep_time_date.hour}:{dep_time_date.minute} <extra></extra>'
        
        
        fig.add_trace(go.Scattermapbox(mode = "lines",
                                        lon = arr_dep_df.stop_lon,
                                        lat = arr_dep_df.stop_lat,
                                        marker = {'color': color_dict[transport_type]},
                                        line={'width':4},
                                        name = f"{transport_type}"))

        fig.add_trace(go.Scattermapbox(mode = "markers",
                                        lon = [x.dep_stop['stop_lon']],
                                        lat = [x.dep_stop['stop_lat']],
                                        showlegend=False,
                                        marker = {'size': 12,'color': color_dict[transport_type]},
                                        hovertemplate = hover_string))
        
        
        i += 1
        
    fig.add_trace(go.Scattermapbox(mode = "markers",
                                        lon = [x.arr_stop['stop_lon']],
                                        lat = [x.arr_stop['stop_lat']],
                                        showlegend=False,
                                        marker = {'size': 12,'color': color_dict[transport_type]},
                                        hovertemplate = f'station name: {x.dep_stop["stop_name"]} <br>arrival time: {arr_time_date.hour}:{arr_time_date.minute} <extra></extra>'))

#     plotly.offline.plot(fig)               #uncomment this if you want the map to open also in a new tab of the browser
    fig.show()

   