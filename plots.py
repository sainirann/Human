import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def build_histogram(data, human_or_robot, color):
    val = np.log(data)
    plt.hist(val, range=(0, val.max()), bins=50, alpha=0.5, color=color)
    plt.title(human_or_robot)
    plt.show()

load_train_info = pd.read_csv(filepath_or_buffer='train_cleaned.csv')

hist_plotter_human = load_train_info.loc[load_train_info['outcome'] == 0.0]
build_histogram(hist_plotter_human['no_of_bids'], 'Bids made by Human', 'mediumblue')
build_histogram(hist_plotter_human['no_of_country_visited'], 'Number of Country Visited by Human Bidder', 'mediumblue')
build_histogram(hist_plotter_human['no_of_device'], 'Number of Device used by Human Bidder', 'mediumblue')
build_histogram(hist_plotter_human['no_of_ip'], 'Number of IP used by Human Bidder', 'mediumblue')
build_histogram(hist_plotter_human['no_of_url'], 'Number of URL used by Human Bidder', 'mediumblue')
build_histogram(hist_plotter_human['no_of_auction_participated'], "Number of Auction Participated by Human Bidder", 'mediumblue')
build_histogram(hist_plotter_human['no_of_auction_won'], "Number of Auction Won by Human Bidder", 'mediumblue')
build_histogram(hist_plotter_human['ip_entropy'], "Entropy of IP by Human Bidder", 'mediumblue')
build_histogram(hist_plotter_human['url_entropy'], "Entropy of URL by Human Bidder", 'mediumblue')

hist_plotter_robot = load_train_info.loc[load_train_info['outcome'] == 1.0]
build_histogram(hist_plotter_robot['no_of_bids'], 'Bids made by Bot', 'red')
build_histogram(hist_plotter_robot['no_of_country_visited'], 'Number of Country', 'red')
build_histogram(hist_plotter_robot['no_of_device'], 'Number of different Device used by Bot', 'red')
build_histogram(hist_plotter_robot['no_of_ip'], 'Number of IP used by Bot', 'red')
build_histogram(hist_plotter_robot['no_of_url'], 'Number of URL used by Bot', 'red')
build_histogram(hist_plotter_robot['no_of_auction_participated'], "Number of Auction Participated by Bot", 'red')
build_histogram(hist_plotter_robot['no_of_auction_won'], "Number of Auction Won by Bot", 'red')
build_histogram(hist_plotter_robot['ip_entropy'], "Entropy of IP by Bot", 'red')
build_histogram(hist_plotter_robot['url_entropy'], "Entropy of URL by Bot", 'red')

load_train_info.drop(['bidder_id', 'outcome'], axis=1, inplace=True)

corr = load_train_info.corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
plt.show()
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
