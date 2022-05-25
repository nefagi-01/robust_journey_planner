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
    def __init__(self, dep_stop, arr_stop, dep_time, arr_time, trip, maximum_delay):
        super().__init__(dep_stop, arr_stop, dep_time, arr_time)
        self.trip = trip
        self.confidence = self.compute_confidence(maximum_delay)

    def get_trip_id(self):
        return self.trip_id

    def get_confidence(self):
        return self.confidence

    def compute_confidence(self, maximum_delay):
        # TODO
        return 1.

    def __str__(self):
        return "At {}, take the line {} with arrival at time {} in {}.".format(datetime.utcfromtimestamp(self.dep_time).strftime('%H:%M:%S'), self.trip["trip_short_name"],
                                                                              datetime.utcfromtimestamp(self.arr_time).strftime('%H:%M:%S'), self.arr_stop['stop_name'])


class Footpath(Link):
    def __init__(self, dep_stop, arr_stop, dep_time, arr_time):
        super().__init__(dep_stop, arr_stop, dep_time, arr_time)

    def __str__(self):
        return "At {}, walk from station {} to station {}, with expected arrival at {} (estimated total time for moving inside stations of {} seconds).".format(datetime.utcfromtimestamp(self.dep_time).strftime('%H:%M:%S'),
                                                                                                self.dep_stop['stop_name'],
                                                                                                self.arr_stop['stop_name'],
                                                                                                datetime.utcfromtimestamp(self.arr_time).strftime('%H:%M:%S'),
                                                                                                CHANGE_TIME)


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
        
    def get_dep_time(self):
        return self.links[0].dep_time
    
    def get_links(self):
        return self.links
    
    def get_confidence(self):
        return self.considence