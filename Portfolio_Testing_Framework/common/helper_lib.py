
import pickle
import plotly.graph_objects as go
#######################################################
# The function calculate FIP (frog-in-the-pan) value
#
#     FIP = sign(Past return)*[% negative - % positive]
#
# The more negative the FIP, the better. 
# The FIP algorithm separates high momentum stocks
# into those that have more continues price path
# (smooth, with a slow diffusion of gradual information
# elements) versus thos high momentum stocks that have 
# more discrete price paths (jumpy, with immidiate 
# information lemets)
#
# input close prices as pandas dataframe
#
# return FIP series
#
#######################################################
def calculateFIP(prices):

    r = prices.pct_change().dropna()
    kk = (prices.iloc[-1].values / (prices.iloc[0].values+0.0000001) - 1)
    past_r_sign = [1 if x > 0 else -1 for x in kk]
    pos_perc = r[r > 0].count() / r.count()
    neg_perc = r[r < 0].count() / r.count()
    return past_r_sign * (neg_perc - pos_perc)

##########################################################
# The function calculates the momentum
#       M(n) = Pt / P(t-n) - 1
# Input: 
#       prices : pandas dataframe
#       start_index : index from the momentum will be calculated
#       momentum_window : index to the momentum will be calculated
# 
# Return : momentum values
##########################################################
def calculateMomentum(prices, start_index, momentum_window, back_offset = 0):
    return (prices.iloc[start_index-back_offset] / prices.iloc[-momentum_window-back_offset] - 1)

##########################################################
# The function calculates the momentum
#       M(n) = (price - MA) / price
# Input: 
#       prices : pandas dataframe
#       start_index : index from the momentum will be calculated
#       momentum_window : index to the momentum will be calculated
# 
# Return : momentum values
##########################################################
def calculateMaMomentum(prices, start_index, momentum_window, back_offset = 0):
    ma = prices.rolling(momentum_window).mean()
    return (prices.iloc[start_index-back_offset]-ma.iloc[start_index-back_offset]) / prices.iloc[start_index-back_offset]




################################################################################

##########################################################
# The function plots assset allocation history
# input is a dataframe with allocations
##########################################################
def plotAllocation(alloc):
    fig = go.Figure()
    for c in alloc.columns:
        fig.add_trace(go.Bar(x=alloc.index.values, y=alloc[c], name=c, ))
    fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'}, bargap=0.0,bargroupgap=0.0)
    fig.update_layout(autosize=False, width=500, height=500,)
    fig.show()


##########################################################
# The function stores dictionary to file
##########################################################
def storeDictionaryToFile(file_path, dic):
    with open(file_path, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pass

##########################################################
# The function reads dictionary form file
##########################################################
def readDictionaryFromFile(file_path):
    with open(file_path, 'rb') as handle:
        dic = pickle.load(handle)
    return dic


