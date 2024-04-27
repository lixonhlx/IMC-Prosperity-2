from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import collections
import copy
import numpy as np
from collections import defaultdict
from math import erf

empty_dict = {
    'AMETHYSTS' : 0,
    'STARFRUIT' : 0,
    'ORCHIDS' : 0,
    'CHOCOLATE' : 0,
    'GIFT_BASKET' :0,
    'ROSES' : 0,
    'STRAWBERRIES' : 0,
    'COCONUT' : 0,
    'COCONUT_COUPON' : 0
}

def def_value():
    return copy.deepcopy(empty_dict)

class Trader:
    def __init__(self):
        self.position = copy.deepcopy(empty_dict)
        self.POSITION_LIMIT = {
            'AMETHYSTS' : 20,
            'STARFRUIT' : 20,
            'ORCHIDS' : 100,
            'STRAWBERRIES' : 350,
            'CHOCOLATE' : 250,
            'ROSES' : 60,
            'GIFT_BASKET' :60, 
            'COCONUT' : 300,
            'COCONUT_COUPON' : 600
        }
        self.volume_traded = copy.deepcopy(empty_dict)
        self.starfruit_cache = []
        self.starfruit_dim = 4
        self.steps = 0
        self.cpnl = defaultdict(lambda : 0)
        self.person_position = defaultdict(def_value)
        self.person_actvalof_position = defaultdict(def_value)
        self.orchids_price = []
        self.random = 0
        
        self.ask_south =[]
        self.bid_south =[]
        self.ask = []
        self.bid = []
        self.importtariff = []
        self.transportfees = []
        self.exporttariff = [] 
        self.orchids_traded = []
        self.cont_buy_basket_unfill = 0
        self.cont_sell_basket_unfill = 0
        self.basket_std = 75.8328 # 117 # 76, 600, 75.8328
        self.implied_vol = []
        
    def calc_next_price_starfruit(self):
        # starfruit cache stores price from 1 day ago, current day resp
        # by price, here we mean mid price

        coef = [-0.01869561,  0.0455032 ,  0.16316049,  0.8090892]
        intercept = 4.481696494462085
        nxt_price = intercept
        for i, val in enumerate(self.starfruit_cache):
            nxt_price += val * coef[i]

        return int(round(nxt_price))
    
    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy==0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val
    
    def compute_orders_amethysts(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        mx_with_buy = -1

        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((self.position[product]<0) and (ask == acc_bid))) and cpos < self.POSITION_LIMIT['AMETHYSTS']:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        mprice_actual = (best_sell_pr + best_buy_pr)/2
        mprice_ours = (acc_bid+acc_ask)/2

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid-1) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask+1)

        if (cpos < self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] < 0):
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid-1), num))
            cpos += num

        if (cpos < self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] > 15):
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid-1), num))
            cpos += num

        if cpos < self.POSITION_LIMIT['AMETHYSTS']:
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.position[product]

        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((self.position[product]>0) and (bid == acc_ask))) and cpos > -self.POSITION_LIMIT['AMETHYSTS']:
                order_for = max(-vol, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if (cpos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] > 0):
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, max(undercut_sell-1, acc_ask+1), num))
            cpos += num

        if (cpos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] < -15):
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, max(undercut_sell+1, acc_ask+1), num))
            cpos += num

        if cpos > -self.POSITION_LIMIT['AMETHYSTS']:
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders
    
    def compute_orders_regression(self, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.position[product]<0) and (ask == acc_bid+1))) and cpos < LIMIT:
                order_for = min(-vol, LIMIT - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.position[product]
        

        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.position[product]>0) and (bid+1 == acc_ask))) and cpos > -LIMIT:
                order_for = max(-vol, -LIMIT-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if cpos > -LIMIT:
            num = -LIMIT-cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders
    
    def calculate_vwap_price(self, order_depth: OrderDepth):
        sell_prices, sell_volumes = zip(*order_depth.sell_orders.items())
        buy_prices, buy_volumes = zip(*order_depth.buy_orders.items())
        sell_volumes = tuple(abs(x) for x in sell_volumes)
        return np.average(sell_prices+buy_prices, weights=sell_volumes+buy_volumes)
    
    def compute_orders_orchids(self, product, state):
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []

        ex_coef = 1 
        im_coef = 1
        transport_coef = 1
        if len(order_depth.sell_orders) == 0 and len(order_depth.buy_orders) == 0:
            return orders, 0
        
        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
        if not product in state.observations.conversionObservations:
            southBid = best_bid
            southAsk = best_ask
            importTariff = 0
            exportTariff = 0
            transportFee = 0
        else:
            southBid = state.observations.conversionObservations[product].bidPrice
            southAsk = state.observations.conversionObservations[product].askPrice
            importTariff = state.observations.conversionObservations[product].importTariff
            exportTariff = state.observations.conversionObservations[product].exportTariff
            transportFee = state.observations.conversionObservations[product].transportFees
        real_bid = southBid - exportTariff - transportFee - 0.1
        real_ask = southAsk + importTariff + transportFee
        conversion = 0
        # Version 3
        if real_bid > best_ask:
            our_ask = best_ask
            our_amount = best_ask_amount #self.position[product] + self.POSITION_LIMIT[product]
            print(f"Case 1: BUY {our_ask} x {our_amount}")
            orders.append(Order(product, our_ask, our_amount))
            conversion += our_amount
        if real_ask < best_ask:
            if best_ask - real_ask > 2:
                our_ask = int(real_ask + 2)
                our_amount = int(best_bid_amount * 2)
                print(f"Case 2: SELL {our_ask} x {our_amount}")
                orders.append(Order(product, our_ask, -our_amount))
                conversion += our_amount  
        if real_ask < best_bid:
            for bid, amount in order_depth.buy_orders.items():
                if bid > real_ask:
                    our_ask = bid
                    our_amount = amount #self.position[product] + self.POSITION_LIMIT[product]
                    print(f"Case 3: SELL {our_ask} x {our_amount}")
                    orders.append(Order(product, our_ask, -our_amount))
                    conversion += our_amount

        
        
        return orders, conversion
    
    def compute_orders_basket(self, order_depth):
        '''STRAWBERRIES : DIP
            CHOCOLATE : BAGUETTE
            ROSES : UKULELE
            GIFT_BASKET : PICNIC_BASKET '''
        
        orders = {'STRAWBERRIES' : [], 'CHOCOLATE': [], 'ROSES' : [], 'GIFT_BASKET' : []}
        prods = ['STRAWBERRIES', 'CHOCOLATE', 'ROSES', 'GIFT_BASKET']
        multiplier = {'STRAWBERRIES' : 6, 'CHOCOLATE': 4, 'ROSES' : 1, 'GIFT_BASKET' : 1}
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell, pb_pos, pb_neg = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            if len(order_depth[p].sell_orders) == 0 or len(order_depth[p].buy_orders) == 0:
                return orders
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            
            pb_pos[p] = self.position[p] + self.POSITION_LIMIT[p] # sell limit
            pb_neg[p] = self.POSITION_LIMIT[p] - self.position[p] # buy limit
            
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
            for price, vol in osell[p].items():
                vol_sell[p] -= vol
        
        print(f"mid_price['GIFT_BASKET']: {mid_price['GIFT_BASKET']},mid_price['STRAWBERRIES']: {mid_price['STRAWBERRIES']}, mid_price['CHOCOLATE']:{mid_price['CHOCOLATE']}, mid_price['ROSES']:{mid_price['ROSES']}")
        
        res = mid_price['GIFT_BASKET'] - mid_price['STRAWBERRIES']*6 - mid_price['CHOCOLATE']*4 - mid_price['ROSES'] - 380
        
        
        # trade like ETF reverse
        trade_at = self.basket_std * 2
        close_at = self.basket_std * (-1)
        if res > trade_at:
            # sell basket
            for price, vol in obuy['GIFT_BASKET'].items():
                vol = min(vol, pb_pos['GIFT_BASKET'])
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', price, -vol))
                pb_pos['GIFT_BASKET'] -= vol
                if pb_pos['GIFT_BASKET'] <= 0:
                    break
        elif res < -trade_at:
            # buy basket
            for price, vol in osell['GIFT_BASKET'].items():
                vol = -min(-vol, pb_neg['GIFT_BASKET'])
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', price, -vol))
                pb_neg['GIFT_BASKET'] += vol
                if pb_neg['GIFT_BASKET'] <= 0:
                    break
        elif res < close_at and self.position['GIFT_BASKET'] < 0:
            # buy basket
            pb_neg_1 = -self.position['GIFT_BASKET']
            for price, vol in osell['GIFT_BASKET'].items():
                vol = -min(-vol, pb_neg_1)
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', price, -vol))
                pb_neg_1 += vol
                if pb_neg_1 <= 0:
                    break
        elif res > -close_at and self.position['GIFT_BASKET'] > 0:
            # sell baskets
            pb_pos_1 = self.position['GIFT_BASKET']
            for price, vol in obuy['GIFT_BASKET'].items():
                vol = min(vol, pb_pos_1)
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', price, -vol))
                pb_pos_1 -= vol
                if pb_pos_1 <= 0:
                    break
        
        # GIFT_BASKET follow Rhianna
        '''if int(round(self.person_position['Rhianna']['GIFT_BASKET'])) > 0:
            if pb_neg['GIFT_BASKET'] > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', best_sell['GIFT_BASKET'], int(pb_neg['GIFT_BASKET'])))
        elif int(round(self.person_position['Rhianna']['GIFT_BASKET'])) < 0:
            if pb_pos['GIFT_BASKET'] > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', best_buy['GIFT_BASKET'], -int(pb_pos['GIFT_BASKET'])))'''
        
        # Chocolate follow Remy
        '''if int(round(self.person_position['Remy']['CHOCOLATE'])) > 0:
            if pb_neg['CHOCOLATE'] > 0:
                orders['CHOCOLATE'].append(Order('CHOCOLATE', best_sell['CHOCOLATE'], int(pb_neg['CHOCOLATE'])))
        elif int(round(self.person_position['Remy']['CHOCOLATE'])) < 0:
            if pb_pos['CHOCOLATE'] > 0:
                orders['CHOCOLATE'].append(Order('CHOCOLATE', best_buy['CHOCOLATE'], int(pb_pos['CHOCOLATE'])))'''
        
        
        return orders
    
    # vol = 0.0101193, T = 250, r = 0
    # vol = 0.01006
    def fun_BS_quick(self, S = 10000, K= 10000, vol = 0.01, T = 250, r = 0, q = 0, ReturnDelta=False):
        d1 = (np.log(S/K) + (r+vol**2/2)*T)/vol/np.sqrt(T)
        d2 = d1 - vol*np. sqrt(T)
        
        normcdf = lambda x: (1.0 + erf(x / np. sqrt (2.0))) / 2.0
        N1 = normcdf(d1)
        N2 = normcdf(d2)
        
        px = S * N1 - K * np.exp((q-r)*T)*N2
        if ReturnDelta:
            return px, N1 
        else:
            return px

    def fun_fit_vol(self, vol_fit = 0.08, S = 9990, px = 620.5, T = 250, step = 0.00001):
        
        for i in range(30):
            px_new = self.fun_BS_quick(S=S, vol=vol_fit, T=T)
            if abs(px_new-px) < 0.01:
                break
            vol_fit = vol_fit + (px-px_new) * step
        return vol_fit
    
    def compute_orders_coconut(self, state):
        orders = {'COCONUT' : [], 'COCONUT_COUPON': []}
        order_depth: OrderDepth = state.order_depths
        prods = ['COCONUT', 'COCONUT_COUPON']
        
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell, pb_pos, pb_neg = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        for p in prods:
            if len(order_depth[p].sell_orders) == 0 or len(order_depth[p].buy_orders) == 0:
                return orders
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))
            
            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            
            pb_pos[p] = self.position[p] + self.POSITION_LIMIT[p] # sell limit
            pb_neg[p] = self.POSITION_LIMIT[p] - self.position[p] # buy limit
        
        S = mid_price['COCONUT']
        # self.calculate_vwap_price(order_depth[prods[0]])
        px = mid_price['COCONUT_COUPON']
        # self.calculate_vwap_price(order_depth[prods[1]])
        vol = 0.01006
        # fair price of coconut coupon
        if int(round(self.person_position['Raj']['COCONUT'])) > 0:
            S *= 1.05
        elif int(round(self.person_position['Raj']['COCONUT'])) < 0:
            S *= 0.95
        px_new, N1 = self.fun_BS_quick(S = S, K= 10000, vol = vol, T = 250, r = 0, q = 0, ReturnDelta=True)
        
        trade_at = 5
        close_at = -2
        print(f"fair_price: {px_new}, price: {px}, Stock_price: {S}")
        
        if px_new - px > trade_at:
            quantity = min(pb_neg['COCONUT_COUPON'], pb_pos['COCONUT']/N1)
            # buy coconut coupon
            orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', worst_sell['COCONUT_COUPON'], int(quantity)))
            # sell coconut
            # orders['COCONUT'].append(Order('COCONUT', worst_buy['COCONUT'], -int(quantity * N1)))
        
        elif px_new - px < -trade_at:
            # sell coconut coupon
            quantity = min(pb_pos['COCONUT_COUPON'], pb_neg['COCONUT']/N1)
            orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', worst_buy['COCONUT_COUPON'], -int(quantity)))
            # buy coconut
            # orders['COCONUT'].append(Order('COCONUT', worst_sell['COCONUT'], int(quantity*N1)))
        
        elif px_new - px < close_at and self.position['COCONUT_COUPON'] < 0:
            # buy coconut coupon
            vol = -self.position['COCONUT_COUPON']
            assert(vol >= 0)
            if vol > 0:
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', worst_sell['COCONUT_COUPON'], vol))
            # sell coconut
            orders['COCONUT'].append(Order('COCONUT', worst_buy['COCONUT'], -int(min(vol * N1, pb_pos['COCONUT']))))
        
        elif px_new - px > -close_at and self.position['COCONUT_COUPON'] > 0:
            # sell coconut coupon
            vol = self.position['COCONUT_COUPON']
            assert(vol >= 0)
            if vol > 0:
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', worst_buy['COCONUT_COUPON'], -vol))
            # buy coconut
            orders['COCONUT'].append(Order('COCONUT', worst_sell['COCONUT'], int(min(vol*N1, pb_neg['COCONUT']))))
       
        # follow Ruby 
        '''if int(round(self.person_position['Ruby']['COCONUT_COUPON'])) > 0:
            val_ord = pb_neg['COCONUT_COUPON']
            if val_ord > 0:
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', best_sell['COCONUT_COUPON'], int(val_ord)))
        if int(round(self.person_position['Ruby']['COCONUT_COUPON'])) < 0:
            val_ord = -(pb_pos['COCONUT_COUPON'])
            if val_ord < 0:
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', best_buy['COCONUT_COUPON'], int(val_ord)))'''
        
        if int(round(self.person_position['Raj']['COCONUT'])) > 0:
            val_ord = pb_neg['COCONUT']
            if val_ord > 0:
                orders['COCONUT'].append(Order('COCONUT', best_sell['COCONUT'], int(val_ord)))
        if int(round(self.person_position['Raj']['COCONUT'])) < 0:
            val_ord = -(pb_pos['COCONUT'])
            if val_ord < 0:
                orders['COCONUT'].append(Order('COCONUT', best_buy['COCONUT'], int(val_ord)))
        return orders
    
    def compute_orders(self, product, acc_bid, acc_ask, state):
        order_depth: OrderDepth = state.order_depths[product]
        if product == "AMETHYSTS":
            return self.compute_orders_amethysts(product, order_depth, acc_bid, acc_ask)
        elif product == "STARFRUIT":
            return self.compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])
        elif product == 'ORCHIDS':
            return self.compute_orders_orchids(product, state)
    
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {
            'AMETHYSTS' : [],
            'STARFRUIT' : [],
            'ORCHIDS' : [],
            'CHOCOLATE' : [],
            'GIFT_BASKET' : [],
            'ROSES' : [],
            'STRAWBERRIES' : [],
            'COCONUT' : [],
            'COCONUT_COUPON' : []
        }
        for key, val in state.position.items():
            self.position[key] = val
        #print()
        #for key, val in self.position.items():
            #print(f'{key} position: {val}')
        
        timestamp = state.timestamp
        
        if len(self.starfruit_cache) == self.starfruit_dim:
            self.starfruit_cache.pop(0)
            
        _, bs_starfruit = self.values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())))
        _, bb_starfruit = self.values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse=True)), 1)
        
        self.starfruit_cache.append((bs_starfruit+bb_starfruit)/2)
        
        INF = 1e9
    
        starfruit_lb = -INF
        starfruit_ub = INF
        
        if len(self.starfruit_cache) == self.starfruit_dim:
            starfruit_lb = self.calc_next_price_starfruit()-1
            starfruit_ub = self.calc_next_price_starfruit()+1
        
        amethysts_lb = 10000
        amethysts_ub = 10000

        acc_bid = {'AMETHYSTS' : amethysts_lb, 'STARFRUIT' : starfruit_lb} # we want to buy at slightly below
        acc_ask = {'AMETHYSTS' : amethysts_ub, 'STARFRUIT' : starfruit_ub} # we want to sell at slightly above

        self.steps += 1
        
        for product in state.market_trades.keys():
            for trade in state.market_trades[product]:
                if trade.buyer == trade.seller:
                    continue
                #if trade.timestamp < state.timestamp - 500:
                    #continue
                # print(f'We are trading {product}, {trade.buyer}, {trade.seller}, {trade.quantity}, {trade.price}')
                self.person_position[trade.buyer][product] = 1.5
                self.person_position[trade.seller][product] = -1.5
                
                self.person_actvalof_position[trade.buyer][product] += trade.quantity
                self.person_actvalof_position[trade.seller][product] += -trade.quantity
        
        for product in ['AMETHYSTS', 'STARFRUIT']:
            orders = self.compute_orders(product, acc_bid[product], acc_ask[product], state)
            result[product] += orders
        
        # round 2 : ORCHIDS
        orders, conversions = self.compute_orders('ORCHIDS', 0, 0, state)
        result['ORCHIDS'] += orders
        print("traderData: " + state.traderData)
        
        print(f"Observations: {state.observations}")
        # round 3
        orders = self.compute_orders_basket(state.order_depths)
        result['GIFT_BASKET'] += orders['GIFT_BASKET']
        result['STRAWBERRIES'] += orders['STRAWBERRIES']
        result['CHOCOLATE'] += orders['CHOCOLATE']
        result['ROSES'] += orders['ROSES']
        
        # round 4
        orders = self.compute_orders_coconut(state)
        result['COCONUT'] += orders['COCONUT']
        result['COCONUT_COUPON'] += orders['COCONUT_COUPON']
        
        #just doing print and calculation
        for product in state.own_trades.keys():
            for trade in state.own_trades[product]:
                if trade.timestamp != state.timestamp - 100:
                    continue
                # print(f'We are trading {product}, {trade.buyer}, {trade.seller}, {trade.quantity}, {trade.price}')
                self.volume_traded[product] += abs(trade.quantity)
                if trade.buyer == "SUBMISSION":
                    self.cpnl[product] -= trade.quantity * trade.price
                else:
                    self.cpnl[product] += trade.quantity * trade.price
        
        totpnl = 0
        
        for product in state.order_depths.keys():
            settled_pnl = 0
            best_sell = min(state.order_depths[product].sell_orders.keys())
            best_buy = max(state.order_depths[product].buy_orders.keys())

            if self.position[product] < 0:
                settled_pnl += self.position[product] * best_buy
            else:
                settled_pnl += self.position[product] * best_sell
            totpnl += settled_pnl + self.cpnl[product]
            #print(f"For product {product}, {settled_pnl + self.cpnl[product]}, {(settled_pnl+self.cpnl[product])/(self.volume_traded[product]+1e-20)}")

        
        # print(f"Timestamp {timestamp}, Total PNL ended up being {totpnl}")
        # print(f'Will trade {result}')
        # print("End transmission")
        
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        conversions = abs(self.position['ORCHIDS'])
        return result, conversions, traderData