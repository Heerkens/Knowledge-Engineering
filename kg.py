import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go

class KnowledgeGraph:
    def __init__(self, df_turn: pd.DataFrame, df_comp: pd.DataFrame, df_countries: pd.DataFrame):
        self.turn = df_turn
        self.comp = df_comp[df_comp.name.isin(self.turn.Company.str.lower())]
        self.countries = df_countries
        self.sample_network()
        self.turn = self.merge_turn_company()

    def sample_network(self, n_employees: int = None):
        """
        Sample a knowledge graph the size of `n_employees` (access at `KnowledgeGraph.network`)
        
        :param n_employees: number of employees to sample (default `None`: sample all employees)
        """
        
        if (n_employees==None) or n_employees > self.turn.shape[0]:
            n_employees = self.turn.shape[0]
        elif n_employees==0:
            n_employees = 1
        
        turn_filt = self.turn.sample(n_employees)
        comp_filt = self.comp[self.comp.name.isin(turn_filt.Company.str.lower())]

        turn_comp_edges = turn_filt[['turn_idx', 'Company']].copy()
        turn_comp_edges['Company'] = turn_comp_edges.Company.str.lower()
        turn_comp_edges.columns = ['source', 'target']
        turn_comp_edges['edge'] = 'works at'
        
        comp_country_edges = comp_filt[['name', 'country']].copy()
        comp_country_edges.columns = ['source', 'target']
        comp_country_edges['edge'] = 'is located in'
        
        comp_industry_edges = turn_filt.groupby('Company').industry.first().to_frame().reset_index()
        comp_industry_edges['Company'] = comp_industry_edges.Company.str.lower()
        comp_industry_edges['industry'] = comp_industry_edges.industry.apply(lambda x: x + ' industry')
        comp_industry_edges.columns = ['source', 'target']
        comp_industry_edges['edge'] = 'active in'

        df_edges = pd.concat([turn_comp_edges, comp_country_edges, comp_industry_edges])
        
        G = nx.from_pandas_edgelist(df_edges, 
                                    source='source', 
                                    target='target', 
                                    edge_attr=True, 
                                    create_using=nx.MultiDiGraph())
        
        G = self.set_network_attributes(G)
        
        self.network = G

    def set_network_attributes(self, network):
        """
        Auxiliary function to set the node attributes of the provided network
        
        :param network: network for which the node attributes have to be set
        :return: network with node attributes
        """
        # set node attributes that are in dataframe
        turn_attrs = self.turn.set_index('turn_idx').to_dict('index')
        nx.set_node_attributes(network, turn_attrs)

        comp_attrs = self.comp.iloc[:,2:].set_index('name').to_dict('index')
        nx.set_node_attributes(network, comp_attrs)

        countries_attr = self.countries.set_index('Country').to_dict('index')
        nx.set_node_attributes(network, countries_attr)
        
        # set the instance type and corresponding layer
        for node, attr in network.nodes(data=True):
            if node[:2]=='e_': # employee
                attr['layer'] = 0
                attr['instance'] = 'employee'
            elif node in ['netherlands', 'france', 'spain', 'italy', 'germany', 'switzerland']: # country
                attr['layer'] = 2
                attr['instance'] = 'country'
            elif node[-8:] == 'industry':
                attr['layer'] = 2
                attr['instance'] = 'industry'
            else: # company
                    attr['layer'] = 1
                    attr['instance'] = 'company'
        return network

    def plot_network_mpl(self, kind='multipartite', *args):
        """
        Visualize the network of the knowledge graph using `matplotlib`
        
        :param kind: layout to use for visualizing (default multipartite, otherwise spring)
        :param *args: any additional arguments for either `nx.multipartite_layout()` or `nx.spring_layout()`
        """
        assert self.network, 'Sample a network using `sample_network` first'
        plt.figure(figsize=(12,12))
        if kind=='multipartite':
            pos = nx.multipartite_layout(self.network, subset_key='layer', *args)
        else:
            pos = nx.spring_layout(self.network, *args)
        color = [['gold', 'violet', 'limegreen'][attr['layer']] for _, attr in self.network.nodes(data=True)]
        nx.draw(self.network, with_labels=False, node_color=color, edge_cmap=plt.cm.Blues, pos = pos)
    
    def plot_network_plotly(self, kind='multipartite', show=True, *args) -> go.Figure:
        """
        Visualize the network of the knowledge graph using `plotly`
        
        :param kind: layout to use for visualizing (default multipartite, otherwise spring)
        :param show: boolean, figure will be shown if set to `True` 
        :param *args: any additional arguments for either `nx.multipartite_layout()` or `nx.spring_layout()`
        :return: the plotly network of class `go.Figure`
        """
        assert self.network, 'Sample a network using `sample_network()` first'

        if kind=='multipartite':
            pos = nx.multipartite_layout(self.network, subset_key='layer', *args)
        else:
            pos = nx.spring_layout(self.network, *args)
        
        color = []
        for _, attr in self.network.nodes(data=True):
            if attr['instance'] == 'employee': # employee
                if attr['turnover'] == 0:
                    color.append('gold')
                else:
                    color.append('black')
            elif attr['instance'] == 'industry':
                color.append('slateblue')
            else: # company or country
                color.append(['violet', 'limegreen'][attr['layer']-1])
        
        pos_x = []
        for n, p in pos.items():
            self.network.nodes[n]['pos'] = p
            pos_x.append(p[0])
        pos_x = np.unique(pos_x)

        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5,color='#888'),
            hoverinfo='none',
            mode='lines')

        for edge in self.network.edges():
            x0, y0 = self.network.nodes[edge[0]]['pos']
            x1, y1 = self.network.nodes[edge[1]]['pos']
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
        
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                color=[],
                size=10,
                line=dict(width=0)))

        node_text = []
        node_color = []
        for node, attr in self.network.nodes(data=True):
            x, y = self.network.nodes[node]['pos']
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            text = '<b>' + str(node) + '</b><br><br>'
            for key, value in attr.items():
                if key not in ['pos', 'layer', 'Company', 'country']:
                    text += '{}: {}<br>'.format(key, value)
            for key, value in attr.items():
                if key in ['Company', 'country']:
                    text += '<br>{}: {}<br>'.format(key, value)
            node_text.append(text)
            node_color.append(attr['instance'])
        node_trace.text = node_text
        node_trace.marker.color = color

        fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        
        fig.add_trace(go.Scatter(
            x=[pos_x[0], pos_x[1], pos_x[2]-0.010],
            y=[1.1, 1.1, 1.1],
            mode="text",
            name="Instance type",
            text=["Employees", "Companies", "Countries/industries"],
            textposition="top center"
        ))
        
        if show:
            fig.show()
        
        return fig

    def merge_turn_company(self, turn_merge=['country', 'size range']):
        """
        Iterates over the employee/company nodes in the graph to add features of corresponding company/employee nodes
        
        :param turn_merge: list of company features to add to employee nodes
        :return: pandas `DataFrame` with employee attributes and the merged company attributes
        """
        lst_employees = []
        for node, attr in self.network.nodes(data=True): 
            if attr['instance']=='employee':
                lst_employees.append(node)
                neighbor = list(self.network.neighbors(node))[0]
                for col in turn_merge:
                    attr[col] = self.network.nodes[neighbor][col]
            elif attr['instance'] == 'company':
                neighbor = list(self.network.predecessors(node))[0]
                attr['industry'] = self.network.nodes[neighbor]['industry']
        merged_turn = pd.DataFrame.from_dict(dict(self.network.subgraph(lst_employees).nodes(data=True)), orient='index')
        merged_turn = merged_turn.reset_index().rename(columns={'index':'turn_idx'})
        return merged_turn
