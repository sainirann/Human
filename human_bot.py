import pandas as pd
import math


class FeatureExtraction:

    def __init__(self):
        self.bid_info_cols = ['bid_id', 'bidder_id', 'auction', 'merchandise', 'device', 'time', 'country', 'ip', 'url']
        self.bid_info = pd.read_csv(filepath_or_buffer='bids.csv', names=self.bid_info_cols, skiprows=1)
        self.bid_info = self.bid_info.astype({'time': 'long'})
        self.bid_info.sort_values(by=['time'], inplace=True)
        self.csv_features = ['bidder_id', 'payment_account', 'address']
        self.output_label = ['outcome']

    def fill_column_na_with_mean(self, data, feature):
        if 'outcome' in data.columns:
            s = data.groupby('outcome')[feature].mean()
            data[feature].fillna(data['outcome'].map(s), inplace=True)
        else:
            s = data[feature].mean()
            data[feature].fillna(s, inplace=True)

    def fill_column_na_with_median(self, t, feature):
        if 'outcome' in t.columns:
            s = t.groupby('outcome')[feature].median()
            t[feature].fillna(t['outcome'].map(s), inplace=True)
        else:
            s = t[feature].median()
            t[feature].fillna(s, inplace=True)

    def calculate_entropy(self, series):
        return series.map(lambda x: -(x * math.log2(x))).sum()

    def extract_train_or_test_features(self, filename_without_extension, load_bid_info, features):
        load_train_info = pd.read_csv(filepath_or_buffer=filename_without_extension + ".csv", names=features, skiprows=1)

        ######### No of bids made by the bidder ##########

        load_train_info = pd.merge(load_train_info, load_bid_info.groupby('bidder_id').agg(
            no_of_bids=('bidder_id', 'count')), how='left', on='bidder_id')
        load_train_info['no_of_bids'].fillna(0, inplace=True)

        ######### One hot encoding of merchandise ##########
        merchandises = load_bid_info['merchandise'].unique()

        load_bid_info_for_merchandise = pd.get_dummies(data=load_bid_info, columns=['merchandise'])
        load_bid_info_for_merchandise.drop(
            load_bid_info_for_merchandise.columns.difference(['merchandise_' + m for m in merchandises] + ['bidder_id']),
            1, inplace=True)

        load_bid_info_for_merchandise = load_bid_info_for_merchandise.groupby(
            ['bidder_id']).agg(lambda s: 1 if s.sum() > 0 else 0)
        load_train_info = pd.merge(load_train_info, load_bid_info_for_merchandise, how='left', on='bidder_id')
        load_train_info[['merchandise_' + m for m in merchandises]].fillna(0, inplace=True)

        ############### No of auction participated, No of country went, No of device/url/ip used by bidder ###########

        no_of_auctionpart_countryvisit_dev_ip_url = load_bid_info.groupby(
            'bidder_id')['auction', 'country', 'device', 'ip', 'url'].nunique()
        no_of_auctionpart_countryvisit_dev_ip_url = no_of_auctionpart_countryvisit_dev_ip_url.rename(
            columns={'auction': 'no_of_auction_participated', 'country': 'no_of_country_visited',
                     'device': 'no_of_device', 'ip': 'no_of_ip', 'url': 'no_of_url'})

        load_train_info = pd.merge(load_train_info, no_of_auctionpart_countryvisit_dev_ip_url, how='left', on='bidder_id')
        load_train_info.fillna(0, inplace=True)

        ################## Mean bids per auction ######################

        load_train_info = pd.merge(load_train_info, load_bid_info.groupby(['bidder_id', 'auction'])
                                   .agg(mean_bids_per_auction=('auction', 'count')).groupby('bidder_id').mean(),
                                   how='left', on='bidder_id')
        load_train_info['mean_bids_per_auction'].fillna(0, inplace=True)

        ################## Number of auction won ##################

        load_train_info = pd.merge(load_train_info,
                                   load_bid_info.groupby('auction').nth(-1).groupby('bidder_id')
                                   .agg(no_of_auction_won=('bidder_id', 'count')), how='left', on='bidder_id')
        load_train_info['no_of_auction_won'].fillna(0, inplace=True)

        #################### Response Time #######################

        load_bid_info = load_bid_info.sort_values(by=['auction', 'time'])
        load_bid_info['resp_time_diff'] = load_bid_info.groupby('auction')['time'].diff()
        dropped_load_bid_info = load_bid_info.dropna()
        resp_time_diff = dropped_load_bid_info.groupby('bidder_id').agg(min_resp_time_diff=('resp_time_diff', 'min'),
                                                                        max_resp_time_diff=('resp_time_diff', 'max'),
                                                                        mean_resp_time_diff=('resp_time_diff', 'mean'),
                                                                        std_resp_time_diff=('resp_time_diff', 'std'),
                                                                        median_resp_time_diff=('resp_time_diff', 'median'))
        load_train_info = pd.merge(load_train_info, resp_time_diff, how='left', on='bidder_id')

        ################## Handled NaN values with mean #################

        self.fill_column_na_with_mean(load_train_info, 'min_resp_time_diff')
        self.fill_column_na_with_mean(load_train_info, 'max_resp_time_diff')
        self.fill_column_na_with_mean(load_train_info, 'mean_resp_time_diff')
        self.fill_column_na_with_mean(load_train_info, 'std_resp_time_diff')
        self.fill_column_na_with_mean(load_train_info, 'median_resp_time_diff')

        ################### Time difference between consecutive bids made by the same bidder ###################

        load_bid_info = load_bid_info.sort_values(by=['bidder_id', 'time'])
        load_bid_info['time_diff_btw_consecutive_bids'] = load_bid_info.groupby('bidder_id')['time'].diff()
        dropped_load_bid_info = load_bid_info.dropna()
        time_diff_consecutive = dropped_load_bid_info.groupby(
            'bidder_id').agg(min_time_diff_consecutive=('time_diff_btw_consecutive_bids', 'min'),
                             max_time_diff_consecutive=('time_diff_btw_consecutive_bids', 'max'),
                             mean_time_diff_consecutive=('time_diff_btw_consecutive_bids', 'mean'),
                             std_time_diff_consecutive=('time_diff_btw_consecutive_bids', 'std'),
                             median_time_diff_consecutive=('time_diff_btw_consecutive_bids', 'median'))
        load_train_info = pd.merge(load_train_info, time_diff_consecutive, how='left', on='bidder_id')

        ################## Handled NaN values with mean #################

        self.fill_column_na_with_mean(load_train_info, 'min_time_diff_consecutive')
        self.fill_column_na_with_mean(load_train_info, 'max_time_diff_consecutive')
        self.fill_column_na_with_mean(load_train_info, 'mean_time_diff_consecutive')
        self.fill_column_na_with_mean(load_train_info, 'std_time_diff_consecutive')
        self.fill_column_na_with_mean(load_train_info, 'median_time_diff_consecutive')

        ############################ Entropy of IP #####################

        prob_of_ip = (load_bid_info.groupby(['bidder_id', 'ip'])['ip'].count() / load_bid_info.groupby(['bidder_id'])['bidder_id'].count()).reset_index(name="probabilities")
        load_train_info = pd.merge(load_train_info, prob_of_ip.groupby(['bidder_id']).agg(ip_entropy=('probabilities', self.calculate_entropy)), how='left', on='bidder_id')
        load_train_info['ip_entropy'].fillna(0, inplace=True)

        ########################## Entropy of URL #####################

        prob_of_url = (load_bid_info.groupby(['bidder_id', 'url'])['url'].count() / load_bid_info.groupby(['bidder_id'])['bidder_id'].count()).reset_index(name="probabilities")
        load_train_info = pd.merge(load_train_info, prob_of_url.groupby(['bidder_id']).agg(url_entropy=('probabilities', self.calculate_entropy)), how='left', on='bidder_id')
        load_train_info['url_entropy'].fillna(0, inplace=True)

        ########################## Bids per auction by a bidder #####################

        bids_price = load_bid_info.groupby(['bidder_id', 'auction']).agg(bids_price_per_auction=('auction', 'count')).groupby('bidder_id').agg(min_bids=('bids_price_per_auction', 'min'),
                                                                         max_bids=('bids_price_per_auction', 'max'),
                                                                         mean_bids=('bids_price_per_auction', 'mean'),
                                                                         median_bids=('bids_price_per_auction', 'std'),
                                                                         std_bids=('bids_price_per_auction', 'median'))

        load_train_info = pd.merge(load_train_info, bids_price, how='left', on='bidder_id')

        ################## Handled NaN values with mean #################

        self.fill_column_na_with_mean(load_train_info, 'min_bids')
        self.fill_column_na_with_mean(load_train_info, 'max_bids')
        self.fill_column_na_with_mean(load_train_info, 'mean_bids')
        self.fill_column_na_with_mean(load_train_info, 'std_bids')
        self.fill_column_na_with_mean(load_train_info, 'median_bids')

        ######################## Bids Price by time #########################

        bids_price_by_time = load_bid_info.groupby('bidder_id').agg(min_bids_price_by_time=('time', 'min'),
                                                                    max_bids_price_by_time=('time', 'max'),
                                                                    mean_bids_price_by_time=('time', 'mean'),
                                                                    std_bids_price_by_time=('time', 'std'),
                                                                    median_bids_price_by_time=('time', 'median'))

        load_train_info = pd.merge(load_train_info, bids_price_by_time, how='left', on='bidder_id')

        ################## Handled NaN values with mean #################

        self.fill_column_na_with_mean(load_train_info, 'min_bids_price_by_time')
        self.fill_column_na_with_mean(load_train_info, 'max_bids_price_by_time')
        self.fill_column_na_with_mean(load_train_info, 'mean_bids_price_by_time')
        self.fill_column_na_with_mean(load_train_info, 'std_bids_price_by_time')
        self.fill_column_na_with_mean(load_train_info, 'median_bids_price_by_time')



        bidder_id = load_train_info['bidder_id']
        load_train_info.drop(['bidder_id', 'payment_account', 'address'], axis=1, inplace=True)

        load_train_info['bidder_id'] = bidder_id
        load_train_info.to_csv(filename_without_extension + '_cleaned.csv', mode='w', index=False)


feature_extraction = FeatureExtraction()
feature_extraction.extract_train_or_test_features(filename_without_extension='train',
                               load_bid_info=feature_extraction.bid_info,
                               features=feature_extraction.csv_features + feature_extraction.output_label)
feature_extraction.extract_train_or_test_features(filename_without_extension='test',
                               load_bid_info=feature_extraction.bid_info,
                               features= feature_extraction.csv_features)
