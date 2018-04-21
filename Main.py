import dota2api
import time
import numpy as np


tempLoad = np.load('traindataNew.npy')
Data = np.ndarray.tolist(tempLoad)
lastCounter = np.load('counter.npy')
#Data = []
#lastCounter = 0
api = dota2api.Initialise("9AD4E20AA4D2C0A43FA90ABB6E4AA5A6")
api.set_raw_mode(1)
numberOfMatches = 1000000
numOfMat = numberOfMatches/100
for i in range(1, int(numOfMat+1)):
    try:
        m = api.get_match_history_by_seq_num(2000000000 + (i * 100) + ((lastCounter + 1) * 100))
    except Exception as ex:
        print("Error Retrying now")
        time.sleep(5)
        continue

    lastCounter = lastCounter + 1
    np.save('counter',lastCounter)
    print(str(i*100) + "/"+ str(numberOfMatches))
    for j in range(0, 100):
        temp = m.get('matches');
        t = temp[j];
        #Add no leaver?
        if(t["lobby_type"] == 0):
            if(t["game_mode"] == 1):
                p = t["players"]
                playersHero = np.zeros([10,1])
                for a in range(0, 10):
                    tempPlayer = p[a]
                    playersHero[a] = tempPlayer["hero_id"]
                Output = np.append(int(t["radiant_win"]), playersHero,int(t["duration"]))
                Data.append(Output)
                D = np.asanyarray(Data)

                np.save('traindataNew', D)
    print('Length Of Data Array', str(len(D)))
#    MatchInfo = MatchInfo + m.get('matches')
    time.sleep(2)

print(np.shape(D))
#trainData = np.memmap('traindataNew.mymemmap',dtype='float16',mode='w+',shape=np.size(D))
#trainData = D
