import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numba  
import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


#plot graph
def plotChartWithSignal(df, day, bar = 0.003, lines_col=[], additiona_ind=[], show_markers=True, show_cross_idx=True, show_only_bars=False, _label_col='dir' ):

    stock = df.copy()
    do=0.07
    dc=0.07

    if show_only_bars == False:
        if show_markers == True:
            f = stock[_label_col]==1.0
            stock.loc[f,"Marker"] = stock["low"]-do
            stock.loc[f,"Symbol"] = 'triangle-up'
            stock.loc[f,"Color"] = "green"

            stock.loc[f,"MarkerC"] = stock["high"]*(1 + bar)+dc
            stock.loc[f,"SymbolC"] = 'triangle-down-open'
            stock.loc[f,"ColorC"] = "green"

            f = stock[_label_col]==-1.0
            stock.loc[f,"Marker"] = stock["high"]+do
            stock.loc[f,"Symbol"] = 'triangle-down'
            stock.loc[f,"Color"] = "red"

            stock.loc[f,"MarkerC"] = stock["low"]-stock["low"]*bar-dc
            stock.loc[f,"SymbolC"] = 'triangle-up-open'
            stock.loc[f,"ColorC"] = "red"


    stock = stock.set_index("date")

    #f = stock[_label_col] != 0
    #stock[f].head(10)

    grp1 = stock.groupby(stock.index).get_group(day)
    grp1 = grp1.fillna(0)


    trace_list = []

    Candle = go.Candlestick(x=grp1.Nr,
                           open=grp1.open,
                           high=grp1.high,
                           low=grp1.low,
                           close=grp1.close
                           )
    trace_list.append(Candle)

    if show_only_bars == False:
        if show_markers == True:
            f = grp1[_label_col] != 0
            Trace = go.Scatter(x=grp1[f].Nr,
                            y=grp1[f].Marker,
                            mode='markers',
                            name ='markers',
                            marker=go.scatter.Marker(size=10,
                                                symbol=grp1[f]["Symbol"],
                                                color=grp1[f]["Color"])
                            )
            trace_list.append(Trace)

            if show_cross_idx == True:
                TraceC = go.Scatter(x=grp1[f].cross_idx,
                                y=grp1[f].MarkerC,
                                mode='markers',
                                name ='markers',
                                marker=go.scatter.Marker(size=10,
                                                    symbol=grp1[f]["SymbolC"],
                                                    color=grp1[f]["ColorC"])
                            )
                trace_list.append(TraceC)


        for lname in lines_col:
            trace_list.append( go.Scatter(
                    x = grp1.Nr,
                    y = grp1[lname],
                    mode = 'lines',
                    name = lname ))


        add_trace = []
        # for lname in additiona_ind:
        #     add_trace.append( go.Scatter(
        #             x = grp1.Nr,
        #             y = grp1[lname],
        #             mode = 'lines',
        #             name = lname ))
        for ind in additiona_ind:
            tmp_lst = []
            for lname in ind:
                tmp_lst.append( go.Scatter(
                        x = grp1.Nr,
                        y = grp1[lname],
                        mode = 'lines',
                        name = lname ))
            add_trace.append(tmp_lst)


    #print('additiona_ind', len(additiona_ind))
    # Build figure
    if(len(additiona_ind) > 0):
        fig = make_subplots(rows=len(additiona_ind)+1, cols=1)
    else:
        fig = go.Figure()
    

    # Build figure
    #fig = go.Figure()
    #fig.add_trace(Candle, row=1, col=1)
    #fig.add_trace(Trace)
    #fig.add_trace(TraceC)
    #fig.add_trace(_Ma)
    fig.add_traces(trace_list)

    if show_only_bars == False:
        for i,ind in enumerate(add_trace):
            for t in ind:
                fig.add_trace(t, row=i+2, col=1)


    fig.update_layout(xaxis_rangeslider_visible=False, width=1000, height=600)
    fig.show()




from  sklearn.metrics import roc_curve, auc
def plotRoc(y, y_proba):

    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    lw = 2
    trace1 = go.Scatter(x=fpr, y=tpr, 
                        mode='lines', 
                        line=dict(color='darkorange', width=lw),
                        name='ROC curve (area = %0.2f)' % roc_auc
                    )

    trace2 = go.Scatter(x=[0, 1], y=[0, 1], 
                        mode='lines', 
                        line=dict(color='navy', width=lw, dash='dash'),
                        showlegend=False)

    layout = go.Layout(title='Receiver operating characteristic example',
                    xaxis=dict(title='False Positive Rate'),
                    yaxis=dict(title='True Positive Rate'))

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.show()


