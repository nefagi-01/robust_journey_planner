from collections import defaultdict
import numpy as np
from datetime import datetime

# CONSTANTS
CHANGE_TIME = 120

class Link:
    def __init__(self, dep_stop, arr_stop, dep_time, arr_time):
        self.dep_stop = dep_stop
        self.arr_stop = arr_stop
        self.dep_time = dep_time
        self.arr_time = arr_time

    def get_dep_stop(self):
        return self.dep_stop

    def get_dep_time(self):
        return self.dep_time

    def get_arr_stop(self):
        return self.arr_stop

    def get_arr_time(self):
        return self.arr_time


class Trip(Link):
    def __init__(self, dep_stop, arr_stop, dep_time, arr_time, trip, confidence):
        super().__init__(dep_stop, arr_stop, dep_time, arr_time)
        self.trip = trip
        self.confidence = confidence

    def get_trip_id(self):
        return self.trip_id

    def get_confidence(self):
        return self.confidence

    def __str__(self):
        return "At {}, take the line {} with arrival at time {} in {}.".format(datetime.utcfromtimestamp(self.dep_time).strftime('%H:%M:%S'), self.trip["trip_short_name"],
                                                                              datetime.utcfromtimestamp(self.arr_time).strftime('%H:%M:%S'), self.arr_stop['stop_name'])


class Footpath:
    def __init__(self, dep_stop, arr_stop, dep_time, duration):
        self.dep_stop = dep_stop
        self.arr_stop = arr_stop
        self.dep_time = dep_time
        self.duration = duration

    def __str__(self):
        if self.dep_stop['stop_name'] == self.arr_stop['stop_name']:
            message = "At {}, walk from the previous platform to the platform of the following line (estimated maximum duration: {} seconds).".format(datetime.utcfromtimestamp(self.dep_time).strftime('%H:%M:%S'),
                                                                                                self.duration)
        else:
            message = "At {}, walk from station {} to station {} (estimated maximum duration: {} seconds).".format(datetime.utcfromtimestamp(self.dep_time).strftime('%H:%M:%S'),
                                                                                                self.dep_stop['stop_name'],
                                                                                                self.arr_stop['stop_name'],
                                                                                                self.duration)
        return message


class Change:
    def __init__(self, stop, previous_trip_id, following_trip_id):
        self.stop = stop
        self.previous_trip_id = previous_trip_id
        self.following_trip_id = following_trip_id

    def __str__(self):
        return "After reaching station {}, change from line {} to line {}.".format(
            self.stop['stop_name'],
            self.previous_trip_id['trip_short_name'],
            self.following_trip_id['trip_short_name'])


class Journey:
    def __init__(self):
        self.links = []
        self.confidence = 1.
        self.num_connections = 0
        self.total_duration_footpaths = 0

    def __len__(self):
        return self.num_connections

    def __str__(self):
        return "\n".join([str(link) for link in self.links])

    def add_trip(self, trip):
        self.links.append(trip)
        self.num_connections += 1
        self.confidence *= trip.get_confidence()

    def add_change(self, change):
        self.links.append(change)

    def add_footpath(self, footpath):
        self.links.append(footpath)
        self.total_duration_footpaths += footpath.duration
        
    def get_dep_time(self):
        return self.links[0].dep_time
    
    def get_links(self):
        return self.links
    
    def get_confidence(self):
        return self.confidence
    
    def get_num_connections(self):
        return self.num_connections
    
    def get_total_duration_footpaths(self):
        return self.total_duration_footpaths