from collections import defaultdict
import numpy as np
from datetime import datetime
from journey import *
import bisect

# CSA HELPERS

def search_next_departure(stop_departures, time):
    '''
        Search the first departure after moment time in the same stop through binary search in the list of S[stop]
    '''
    start = 0
    end = len(stop_departures) - 1

    next_departure_index = -1
    while start <= end:
        mid = (start + end) // 2;

        # Search in the right side
        if (stop_departures[mid]['departure_time'] < time):
            start = mid + 1

        # Search in the left side
        else:
            next_departure_index = mid
            end = mid - 1

    # Search is always successful because of the last np.inf departure
    next_profile_entry = stop_departures[next_departure_index]

    return next_departure_index, next_profile_entry

def search_last_connection(connections, max_arrival_time):
    increasing_dep_times = [x[2] for x in connections[::-1]]
    
    i = bisect.bisect_right(increasing_dep_times, max_arrival_time)
    increasing_last_index = i - 1
    
    last_index = len(connections) - increasing_last_index
    
    return last_index


def shift(vector, include_earliest_arrival=False):
    new_vector = vector.copy()
    new_vector[1:] = vector[:-1]
    new_vector[0] = np.inf
    if include_earliest_arrival:
        new_vector[-1] = min(vector[-1], vector[-2])
    return new_vector


def minimize(vector1, vector2):
    return np.minimum(vector1, vector2)


def minimize_with_stops(old_arrivals, old_connections, tau_c, connection):
    indices_to_change = np.where(tau_c < old_arrivals)[0]

    new_arrival_times = old_arrivals.copy()
    new_arrival_times[indices_to_change] = tau_c[indices_to_change]
    new_connections = old_connections.copy()
    for index in indices_to_change:
        new_connections[index] = connection
    return (new_arrival_times, new_connections)


def minimize_with_enter_exit_connections(old_arrivals, old_enter_connections, tau_c, connection, old_exit_connections,
                                         trip_exit_connections):
    indices_to_change = np.where(tau_c < old_arrivals)[0]

    new_arrival_times = old_arrivals.copy()
    new_arrival_times[indices_to_change] = tau_c[indices_to_change]
    new_enter_connections = old_enter_connections.copy()
    new_exit_connections = old_exit_connections.copy()
    for index in indices_to_change:
        new_enter_connections[index] = connection
        new_exit_connections[index] = trip_exit_connections[index]

    return (new_arrival_times, new_enter_connections, new_exit_connections)


class JourneyPlanner:
    def __init__(self, timetable):
        self.stops, self.connections, self.trips, self.footpaths = timetable

    def get_query_connections(self, day, max_arrival_time):
        index_day = day + 5 # skip first 5 attributes from tuple
        connections = [(connection[0], connection[1], connection[2], connection[3], connection[4]) for connection in self.connections if connection[index_day]]
        return connections[search_last_connection(connections, max_arrival_time):]

    def CSA(self, day, target_stop, min_departure_time, max_arrival_time, max_changes, include_earliest_arrival):
        '''
            Implementation of Pareto Connection Scan profile algorithm with interstop footpaths (https://arxiv.org/pdf/1703.05997.pdf)
        '''

        # Get timetable of the day
        connections = self.get_query_connections(day, max_arrival_time)
        footpaths = self.footpaths

        # If max_changes is X, we want X+1 long vectors (one per changes, including max_changes)
        max_changes += 1

        # If including the earliest arrival, increase max changes by one
        if include_earliest_arrival:
            max_changes += 1

        # prepare S, T and pre-process D
        S = defaultdict(lambda: [{'departure_time': np.inf, 'arrival_times': np.ones(max_changes) * np.inf,
                                  'enter_connections': [None] * max_changes, 'exit_connections': [
                                                                                                     None] * max_changes}])  # entries of the type       stop: [(departure_time, [arrival_time_with_1_connection, ..., arrival_time_with_max_changes], incoming_connection, [outgoing_connections])]
        T = defaultdict(lambda: {'arrival_times': np.ones(max_changes) * np.inf, 'exit_connections': [
                                                                                                         None] * max_changes})  # trip: [arrival_time_with_1_connection, ..., arrival_time_with_max_changes], incoming_connection
        D = defaultdict(lambda: np.inf)

        # initialize footpaths from stops to target stop
        if target_stop in footpaths:
            for footpath in footpaths[target_stop].items():
                f_dep_stop, f_dur = footpath
                D[f_dep_stop] = f_dur

        for connection in connections:
            c_dep_stop, c_arr_stop, c_dep_time, c_arr_time, c_trip = connection


            # Check end condition: if the departure time of the connection is before the minimum departure time, we can stop
            if c_dep_time < min_departure_time:
                break

            # Initialize exit connections of trip with the last temporal connection for this trip: after this trip, it is necessary to exit the line as it does not lead anywhere
            # Otherwise, if the current connection leads to the target station, set this as exit connection: does not make sense to go somewhere else after arriving here for the subsequent connections
            if T[c_trip]['exit_connections'][0] is None:# or c_arr_stop == target_stop:
                T[c_trip]['exit_connections'] = [connection] * max_changes
                

            # PHASE 1: FIND tau_c (arrival times given this connection)

            # Arrival times by walk from c_arr_stop / 0 if the arrival stop is target stop
            tau_1 = np.ones(max_changes) * (c_arr_time if c_arr_stop == target_stop else c_arr_time + D[c_arr_stop])

            # Arrival times by continuing on the same trip
            tau_2 = T[c_trip]['arrival_times']

            # Arrival times by moving to another trip
            tau_3 = search_next_departure(S[c_arr_stop], c_arr_time)[1]['arrival_times']
            tau_3 = shift(tau_3, include_earliest_arrival)

            # Find minimum per number of changes between tau_1, tau_2 and tau_3
            tau_c = minimize(minimize(tau_1, tau_2), tau_3)
            

            # PHASE 2: UPDATE S

            # arrivals of earliest departure after c_dep_time from stop c_dep_stop
            next_departure_index, next_profile_entry = search_next_departure(S[c_dep_stop], c_dep_time)
            y = next_profile_entry['arrival_times']
            y_departure_connections = next_profile_entry['enter_connections']
            y_exit_connections = next_profile_entry['exit_connections']
            new_min_arrivals, new_enter_connections, new_exit_connections = minimize_with_enter_exit_connections(y,
                                                                                                                 y_departure_connections,
                                                                                                                 tau_c,
                                                                                                                 connection,
                                                                                                                 y_exit_connections,
                                                                                                                 T[
                                                                                                                     c_trip][
                                                                                                                     'exit_connections'])

            # If the new candidate profile ix not dominated
            if not np.array_equal(y, new_min_arrivals):
                # Update for the departure stop of the connection (as if there was a self-loop in footpaths)
                S[c_dep_stop].insert(next_departure_index,
                                     {'departure_time': c_dep_time, 'arrival_times': new_min_arrivals,
                                      'enter_connections': new_enter_connections,
                                      'exit_connections': new_exit_connections})
                
                # If c_dep_stop is not reachable from any other stop
                if c_dep_stop not in footpaths:
                    continue

                # Update for the footpath departure stop of all footpaths towards the departure stop of the connection
                for footpath in footpaths[c_dep_stop].items():
                    f_dep_stop, f_dur = footpath
                    next_departure_index, next_profile_entry = search_next_departure(S[f_dep_stop],
                                                                                     c_dep_time - f_dur - CHANGE_TIME)  # We assume that we need CHANGE_TIME seconds to move in addition to the duration of the footpath
                    y = next_profile_entry['arrival_times']
                    y_departure_connections = next_profile_entry['enter_connections']
                    y_exit_connections = next_profile_entry['exit_connections']
                    new_min_arrivals, new_enter_connections, new_exit_connections = minimize_with_enter_exit_connections(
                        y, y_departure_connections, tau_c, connection, y_exit_connections,
                        T[c_trip]['exit_connections'])
                    S[f_dep_stop].insert(next_departure_index,
                                         {'departure_time': c_dep_time, 'arrival_times': new_min_arrivals,
                                          'enter_connections': new_enter_connections,
                                          'exit_connections': new_exit_connections})

        # PHASE 3: UPDATE T

        new_min_arrivals, new_exit_connections = minimize_with_stops(T[c_trip]['arrival_times'],
                                                                     T[c_trip]['exit_connections'], tau_c,
                                                                     connection)
        T[c_trip]['arrival_times'] = new_min_arrivals
        T[c_trip]['exit_connections'] = new_exit_connections
            
        
        # Delete list from memory
        del connections

        return S

    def extract_paths_with_k_changes(self, S, source_stop, source_time, target_stop, k):
        assert source_stop != target_stop
        
        paths = set()
        
        profiles_source_stop = S[source_stop]
        first_departure_index, _ = search_next_departure(profiles_source_stop, source_time)

        for i in range(first_departure_index, len(profiles_source_stop)):
            profile_entry = profiles_source_stop[i]
            enter_connection = profile_entry['enter_connections'][k]
            exit_connection = profile_entry['exit_connections'][k]
            current_trip = (enter_connection, exit_connection)

            # If it is not possible to reach target from here in the selected number of changes
            if enter_connection is not None:

                # If we cannot do more changes
                if k == 0:
                    # If this connection or this line is directed to the target stop
                    if enter_connection[1] == target_stop or exit_connection[1] == target_stop:
                        paths.add(current_trip)

                # If we can do more changes and we need to exit the line at some point after this connection
                else:
                    next_source_stop = exit_connection[1]
                    next_source_time = exit_connection[3]
                    
                    # If the budget of changes is not zero and the next stop is the arrival stop, we end
                    if next_source_stop == target_stop:
                        print("wooops")
                        continue
    
                        
                    # If enter_connections and exit_connections do not belong to the same trip, 
                    if exit_connection[4] != enter_connection[4]:
                        print("WHY")
                        # Add to the next source time the duration of the footpath
                        next_source_time += self.footpaths[exit_connection[0]][enter_connection[1]] + CHANGE_TIME
                        
                    
                    paths_from_next = self.extract_paths_with_k_changes(S, next_source_stop, next_source_time,
                                                                        target_stop, k - 1)
                    paths_from_current = [current_trip + path for path in paths_from_next]
                    paths.update(paths_from_current)

        return paths

    def extract_paths_with_at_most_k_changes(self, S, source_stop, source_time, target_stop, k):
        paths = set().union(
            *[self.extract_paths_with_k_changes(S, source_stop, source_time, target_stop, k) for k in range(k + 1)])
        print(paths)
        return paths

    def extract_all_paths(self, S, source_stop, source_time, target_stop):
        profiles_source_stop = S[source_stop]
        first_departure_index, _ = search_next_departure(profiles_source_stop, source_time)

        paths = set()

        for i in range(first_departure_index, len(profiles_source_stop)):
            profile_entry = profiles_source_stop[i]
            enter_exit_connections = set(zip(profile_entry['enter_connections'], profile_entry['exit_connections']))
            for current_trip in enter_exit_connections:
                enter_connection, exit_connection = current_trip
                # If it is not possible to reach target from here
                if enter_connection is None:
                    continue
                # If this connection or this line is directed to the target stop
                if enter_connection[1] == target_stop or exit_connection[1] == target_stop:
                    paths.add(current_trip)
                # If we need to exit the line at some point after this connection
                else:
                    next_source_stop = exit_connection[1]
                    next_source_time = exit_connection[3]
                    paths_from_next = self.extract_all_paths(S, next_source_stop, next_source_time, target_stop)
                    paths_from_current = [current_trip + path for path in paths_from_next]
                    paths.update(paths_from_current)

        return paths

    def process_path(self, source_stop, target_stop, path, maximum_arrival_time):
        '''
            From paths (list of pairs of enter_connection, exit_connections) to journeys (list of links)
        '''

        if path is None:
            return None
        
        print(path)
    
        journey = Journey()
        
        assert len(path) % 2 == 0
        path_len = len(path) // 2
        
        

        # Add first link
        trip_start = path[0]
        trip_end = path[1]
        
        # Handle source_stop != first trip departure stop (=> add footpath from source_stop to first trip departure stop)  
        if source_stop != trip_start[0]:
            footpath = Footpath(self.stops[source_stop], self.stops[trip_start[0]], trip_start[2] - self.footpaths[source_stop][trip_start[0]],
                                    trip_start[2] + CHANGE_TIME)
            journey.add_footpath(footpath)
        
        
        
        next_departure_time = path[2][2] if len(path) > 2 else maximum_arrival_time
        maximum_delay = next_departure_time - trip_end[3]
        trip = Trip(dep_stop=self.stops[trip_start[0]], arr_stop=self.stops[trip_end[1]], dep_time=trip_start[2], arr_time=trip_end[3],
                        trip=self.trips[trip_start[4]], maximum_delay=maximum_delay)
        journey.add_trip(trip)

        # Update previous exit connection for detecting footpaths
        previous_exit_connection = path[1]

        # Add all the other links
        for i in range(1, path_len):
            trip_start = path[i * 2]
            trip_end = path[i * 2 + 1]

            # If the last arrival stop is not the following departing stations, we need to walk from one stop to the other
            if trip_start[0] != previous_exit_connection[1]:
                footpath = Footpath(self.stops[previous_exit_connection[1]], self.stops[trip_start[0]], previous_exit_connection[3] + CHANGE_TIME/2,
                                    self.footpaths[trip_start[0]][previous_exit_connection[1]] - CHANGE_TIME/2)
                journey.add_footpath(footpath)

            # Otherwise, we just need to change in the same station before the new link if the departure station is not
            # the first one
            else:
                change = Change(self.stops[trip_start[0]], self.trips[previous_exit_connection[4]], self.trips[trip_start[4]])
                journey.add_change(change)

            # Finally, add the new trip after the footpath / change
            next_departure_time = path[(i + 1) * 2][2] if len(path) > (i + 1) * 2 else maximum_arrival_time
            maximum_delay = next_departure_time - trip_end[3]
            trip = Trip(dep_stop=self.stops[trip_start[0]], arr_stop=self.stops[trip_end[1]], dep_time=trip_start[2], arr_time=trip_end[3],
                        trip=self.trips[trip_start[4]], maximum_delay=maximum_delay)
            journey.add_trip(trip)

            # Update previous exit connection for detecting footpaths
            previous_exit_connection = trip_end
        print("NEW JOURNEY")
        print(journey)
        last_link = journey.get_links()[-1]
        
        if type(last_link) is Trip or type(last_link) is Footpath:
            print(last_link.dep_stop)
            
        # Handle target_stop != last stop (=> add footpath from last station to destination)    
        if previous_exit_connection[1] != target_stop:
            print(self.footpaths[target_stop])
            print(previous_exit_connection[1])
            footpath = Footpath(self.stops[previous_exit_connection[1]], self.stops[target_stop], previous_exit_connection[3] + CHANGE_TIME/2,
                                    self.footpaths[target_stop][previous_exit_connection[1]] + previous_exit_connection[3] - CHANGE_TIME/2)
            journey.add_footpath(footpath)
        
        
        return journey
    
    def plan_route(self, day, source_stop, target_stop, min_departure_time, max_arrival_time, max_changes=None):

        include_earliest_arrival = max_changes is None

        # If no limit for max_changes, set an arbitrary number for the algo: even with this, we can still have paths of more than max_changes changes
        if max_changes is None:
            max_changes = 10

        # Run CSA
        S = self.CSA(day=day, target_stop=target_stop, min_departure_time=min_departure_time, max_arrival_time=max_arrival_time, max_changes=max_changes, include_earliest_arrival=include_earliest_arrival)
        
        # Extract the paths
        paths = self.extract_all_paths(S, source_stop, min_departure_time,
                                       target_stop) if include_earliest_arrival else self.extract_paths_with_at_most_k_changes(
            S, source_stop, min_departure_time, target_stop, max_changes)
        paths = sorted(paths, key=lambda journey: journey[0][2], reverse=True)
        
        journeys = [self.process_path(source_stop, target_stop, path, max_arrival_time) for path in paths]
        
        # Search journeys of just one footpath
        if target_stop in self.footpaths:
            if source_stop in self.footpaths[target_stop]:
                footpath_journey = Journey()
                footpath_journey.add_footpath(Footpath(self.stops[source_stop], self.stops[target_stop], max_arrival_time - self.footpaths[target_stop][source_stop], max_arrival_time))
                journeys.append(footpath_journey)
            
        # Sort journeys by departure time
        journeys = sorted(journeys, key=lambda journey: journey.get_dep_time(), reverse=True)
        
        for i, journey in enumerate(journeys):
            print("JOURNEY", i + 1)
            print(str(journey))
            
        return journeys