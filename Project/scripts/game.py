import numpy as np


class Game:
    
    # evaluate according to different rules the winner of the game
    def __init__(self,df):
        self.nb_rounds,_ = df.shape
        self.rounds = df
        
        self.m = {'J':10,'Q':11,'K':12}
        for i in range(10):
            self.m[str(i)] = i
    
    def _round_decide(self,rd,rule):
        # rd list [P1,P2,P3,P4,D]
        if rule.strip() == 'standard':
            s = np.array([self.m[card[0]] for card in rd[:4]])
            return np.where((np.max(s) == s) == True)[0] #first tuple
        
        elif rule.strip() == 'advanced':
            D = int(rd[4]) - 1
            sym = rd[D][1]
            
            s = np.array([self.m[card[0]] if card[1]==sym else -1 for card in rd[:4]])
            return [np.argmax(s)] #rule staed no two winner
        else:
            return 
    def gamewinner(self,rule='standard'):
        count = [0] * 4 #four players count
        for i in range(self.nb_rounds):
            rd = list(self.rounds[['P1','P2','P3','P4','D']].iloc[i])
            winners = self._round_decide(rd,rule)
            for win in winners:
                count[win] = count[win] + 1
        return count 
    