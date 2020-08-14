import networkx as nx
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import *
import pandas as pd


class TemporalGraph():
    def __init__(self, data, time_granularity):
        '''

        :param data: DataFrame- source, target, time, weight columns
        :param time_granularity: 'day', 'week', 'month', 'year' or 'hour'
        '''
        data['day'] = data['time'].apply(lambda timestamp: (datetime.utcfromtimestamp(timestamp)).day)
        data['week'] = data['time'].apply(
            lambda timestamp: (datetime.utcfromtimestamp(timestamp)).isocalendar()[1])
        data['month'] = data['time'].apply(lambda timestamp: (datetime.utcfromtimestamp(timestamp)).month)
        data['year'] = data['time'].apply(lambda timestamp: (datetime.utcfromtimestamp(timestamp)).year)
        data['hour'] = data['time'].apply(lambda timestamp: (datetime.utcfromtimestamp(timestamp)).hour)
        if 'weight' not in data.columns:
            data['weight'] = 1
        self.data = data
        self.time_granularity = time_granularity
        self.time_columns, self.step = self._get_time_columns(time_granularity)
        self.static_graph = self.get_static_graph()

    def get_static_graph(self):
        g = nx.from_pandas_edgelist(self.data, source = 'source', target = 'target', edge_attr = ['weight'],
                                    create_using = nx.Graph())
        self.nodes = g.nodes()
        return g

    def filter_nodes(self, thresh = 5):
        nodes2filter = [node for node, degree in self.static_graph.degree() if degree < thresh]
        return nodes2filter

    def get_temporal_graphs(self, min_degree, mode = 'dynamic'):
        '''

        :param filter_nodes: int.  filter nodes with degree<min_degree in all time steps
        :param mode: if not 'dynamic', add all nodes to the current time step without edges
        :return: dictionary. key- time step, value- nx.Graph
        '''
        G = {}
        for t, time_group in self.data.groupby(self.time_columns):
            time_group = time_group.groupby(['source', 'target'])['weight'].sum().reset_index()
            g = nx.from_pandas_edgelist(time_group, source = 'source', target = 'target', edge_attr = ['weight'],
                                        create_using = nx.Graph())
            if mode != 'dynamic':
                g.add_nodes_from(self.nodes)
            g.remove_nodes_from(self.filter_nodes(min_degree))
            G[self.get_date(t)] = g
        self.graphs = G
        return G

    def get_date(self, t):
        time_dict = dict(zip(self.time_columns, t))
        if self.time_granularity == 'hours':
            return datetime(year = time_dict['year'], month = time_dict['month'], day = time_dict['day'],
                            hour = time_dict['hour'])
        elif self.time_granularity == 'days':
            return datetime(year = time_dict['year'], month = time_dict['month'], day = time_dict['day'])
        elif self.time_granularity == 'months':
            return datetime(year = time_dict['year'], month = time_dict['month'], day = 1)
        elif self.time_granularity == 'weeks':
            date_year = datetime(year = time_dict['year'], month = 1, day = 1)
            return date_year + timedelta(days = float((time_dict['week'] - 1) * 7))
        elif self.time_granularity == 'years':
            return datetime(year = time_dict['year'], month = 1, day = 1)
        else:
            raise Exception("not valid time granularity")

    @staticmethod
    def _get_time_columns(time_granularity):
        if time_granularity == 'hours':
            group_time = ['year', 'month', 'day', 'hour']
            step = timedelta(hours = 1)
        elif time_granularity == 'days':
            group_time = ['year', 'month', 'day']
            step = timedelta(days = 1)
        elif time_granularity == 'weeks':
            group_time = ['year', 'week']
            step = timedelta(weeks = 1)
        elif time_granularity == 'months':
            group_time = ['year', 'month']
            step = relativedelta(months = 1)
        elif time_granularity == 'years':
            group_time = ['year']
            step = relativedelta(years = 1)
        else:
            raise Exception("not valid time granularity")
        return group_time, step


if __name__ == "__main__":
    path = r"data/enron/out.enron"
    df = pd.read_table(path, sep = ' ', header = None)
    df.columns = ['source', 'target', 'weight', 'time']
    temporal_graphs = TemporalGraph(data = df, time_granularity = 'weeks')
    graphs = temporal_graphs.get_temporal_graphs(min_degree = 10)
    print(graphs)
