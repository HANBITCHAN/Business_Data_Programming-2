import psycopg2
import pandas as pd
import networkx as nx
import numpy as np
import collections
import matplotlib.pyplot as plt

def filter_event_log(event_log, z):
    """
    Filter the first z events for cases in the log
    (and delete the events of cases with less than z events)
    :param event_log: a df containing the event log
    :param z: the number of events to filter
    :return: a df containing the filtered event log
    """
    event_log = event_log
    rio = []
    grouped = event_log.groupby('caseid')
    for name, group in grouped:
        if len(group) >= z:
            rio.append(group.iloc[:z])
    df = pd.concat(rio)
    df = df.sort_values('ts', ascending=[True])
    df = df.reset_index(drop=True)
    print(df)
    return df

    pass

def get_social_network_handoffs(event_log, z, x):
    """
    Returns a (networkx) graph containing the
    work handoff social network of an event log
    :param event_log: a df containing the event log
    :param z: the number of events to filter
    :param x: the threshold of handoffs to consider edges
    :return: a social network of handoffs graph
    """
    event_log = filter_event_log(event_log, z)
    grouped = event_log.groupby(['caseid'])
    g1 = nx.DiGraph()
    for name, group in grouped:
        activity_list = list(group['activity'])
        resource_list = list(group['resource'])
        edge = []
        hand_off = 0
        compare = activity_list[0]
        compare2 = resource_list[0]
        for i in range(len(resource_list)):
            if compare == activity_list[i]:
                hand_off +=1
            else:
                if compare2 == resource_list[i]:
                    hand_off = 0
                    del compare
                    del compare2
                    compare = activity_list[i]
                    compare2 = resource_list[i]
                else:
                    for j in range(hand_off):
                        edge.append((compare2, resource_list[i]))
                    hand_off = 0
                    del compare
                    del compare2
                    compare = activity_list[i]
                    compare2 = resource_list[i]
        result = collections.Counter(edge)
        for i in range(len(list(result.keys()))):
            if list(result.values())[i] >= x:
                g1.add_nodes_from([list(result.keys())[i][0], list(result.keys())[i][1]])
                g1.add_edge(list(result.keys())[i][0], list(result.keys())[i][1], weight=list(result.values())[i])
    node = list(g1.nodes)
    rio = []
    for i in range(len(node)):
        for j in range(1,len(node)):
            if nx.has_path(g1, node[i], node[j]) == False:
                rio.append((node[i], node[j]))
    print(rio)
    return g1
    pass

def preprocess_event_log(event_log, z):
    """
    Returns a dataframe containing the event log
    filtered and preprocessed to create items
    :param event_log: a df containing the event log
    :param z: the number of events to filter
    :return: a df containing the filtered and preprocessed log
    """
    event_log = filter_event_log(event_log,z)
    groups = event_log.groupby('activity', as_index=True)
    rio = []
    count = 0
    for case, group in groups:
        print("processing ...{0}/{1}".format(count, len(groups)))
        count +=1
        group = group.reset_index(drop=True)
        group2 = group.copy()
        for i in range(len(group)):
            if i == 0:
                group.loc[i, 'ts'] = 0
            else:
                group.loc[i, 'ts'] = (group2.loc[i, 'ts'] - group2.loc[i - 1, 'ts']).seconds
        mean = np.mean(list(group.loc[:,'ts']))
        mean1 = np.mean(list(group.loc[:, 'reqamount']))
        for i in range(len(group)):
            if group.loc[i, 'ts'] < 0.4 * mean:
                group.loc[i, 'ts'] = 'SHORT'
            elif group.loc[i, 'ts'] < 0.65 * mean:
                group.loc[i, 'ts'] = 'MEDIUM'
            else:
                group.loc[i, 'ts'] = 'LONG'
            if group.loc[i, 'reqamount'] < 0.4*mean1:
                group.loc[i, 'reqamount'] = 'SMALL'
            elif group.loc[i, 'reqamount'] < 1.2*mean1:
                group.loc[i, 'reqamount'] = 'MEDIUM'
            else:
                group.loc[i, 'reqamount'] = 'LARGE'
        rio.append(group)


    new_log = pd.concat(rio)
    new_log = new_log.sort_values('id', ascending=[True])
    new_log = new_log[['caseid', 'apptype', 'activity', 'resource', 'reqamount', 'ts']].reset_index(drop=True)
    new_log = pd.get_dummies(new_log, columns = ['apptype', 'activity', 'resource', 'reqamount', 'ts'])
    grouped = new_log.groupby('caseid')
    rio = []
    for name, group in grouped:
        group = group.reset_index(drop=True)
        new_data = pd.DataFrame()
        for i in range(z):
            if i == 0:
                dict_1 = group.iloc[0].to_dict()
                new_data = pd.DataFrame.from_dict([dict_1])
            else:
                dict_1 = group.iloc[i].to_dict()
                new_data2 = pd.DataFrame.from_dict([dict_1])
                new_data = pd.merge(new_data, new_data2, on='caseid', suffixes=('_{0}'.format(i),'_{0}'.format(i+1)))
        rio.append(new_data)
    df = pd.concat(rio)
    df = df.reset_index(drop=True)
    print(df.head(10))
    return df

    pass

def get_item(event_log, z, case_id, attr_name, attr_value, event_pos):
    """
    returns the item list associated with case_id
    :param event_log: a df containing the event log
    :param z: the number of events to filter
    :param case_id: the id of the case to consider
    :param attr_name: name of the attribute (e.g. "resource")
    :param attr_value: the value of the attribute (e.g. "user_34")
    :param event_pos: the position in the case of the event (events always ordered by timestamp, from earlier to later), NOT RELEVANT FOR CASE-lEVEL ATTRIBUTES
    :return: a list of items for case case_id
    """

    event_log = preprocess_event_log(event_log,z)
    if '{0}_{1}_{2}'.format(attr_name, attr_value, event_pos) not in event_log.columns:
        print('There is no {0}_{1}'.format(attr_value, event_pos))
    else:
        index = 0
        for i in range(len(event_log)):
            case_id1 = event_log.loc[i, 'caseid']
            if case_id1 == case_id:
                index = i
        value = event_log.loc[index, '{0}_{1}_{2}'.format(attr_name, attr_value, event_pos)]
        print('result : {0}'.format(value))
    pass
def display_and_save(G, file_name, layout = "fg"):
    """
    This function is given and allows you to (i) display a graph using matplotlib and (ii) save the graph
    is a png file named "file_name"
    :param G: the graph
    :param file_name: the name of the file, e.g. "graph" will save the image in a file named "graph.png"
    :param layout: the layout chosen to visualise the graph (default is  fruchterman_reingold_layout)
    """

    if layout == "spring":
       pos = nx.spring_layout(G)
    elif layout == "shell":
       pos = nx.shell_layout(G)
    elif layout == "spectral":
       pos = nx.spectral_layout(G)
    else:
       pos = nx.fruchterman_reingold_layout(G)

    # nodes
    nx.draw_networkx_nodes(G, pos,
                           nodelist=G.nodes(),
                           node_color='r',
                           node_size=500,
                           alpha=0.8)

    nx.draw_networkx_edges(G, pos,
                           edgelist=G.edges(),
                           width=2, alpha=0.5, edge_color='r')

    labels = {}
    i = 0
    for node in G.nodes():
        labels[node] = str(node)

    nx.draw_networkx_labels(G, pos, labels, font_size=16)

    plt.axis('off')
    plt.savefig(file_name + ".png")  # save as png
    plt.show()  # display

if __name__ == '__main__':
    connection = psycopg2.connect(user="myuser064",
                                  password="064",
                                  host="114.70.14.56",
                                  port="10051",
                                  database="mydb")
    print("---Connecting to database...")
    cursor = connection.cursor()
    print("---done.\n")

    print("Running query 1...\n")
    query1 = "select * from {0}".format('loans')
    query1_output = query1
    cursor.execute(query1_output, ('loans', id))
    records1 = cursor.fetchall()

    col_names = []
    for desc in cursor.description:
        col_names.append(desc[0])
    cursor.close()

    event_log = pd.DataFrame.from_records(list(records1), columns=col_names)
    preprocess_event_log(event_log,13)
    graph = get_social_network_handoffs(event_log,13,3)
    display_and_save(graph, 'ko.png')

    pass