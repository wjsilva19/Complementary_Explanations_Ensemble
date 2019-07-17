import numpy as np
import pandas as pd
import cv2
import os

"""
filenames = sorted(list(os.walk('imgs'))[0][2])

old = [f for f in filenames if len(f) == len('001a.jpg')]
new = [f for f in filenames if len(f) != len('001a.jpg')]

for id_, i in enumerate(old + new):
    img = cv2.imread(os.path.join('imgs', i))
    cv2.imwrite('corrected/%03d.jpg' % id_, img)
"""

df = pd.read_excel('Features_for_interpretability_consensus.xlsx')
df.drop('Image', axis=1, inplace=True)

for label in range(0, 3):
    df_aux = df.copy()
    df_aux.Consensus = (df.Consensus > label + 1).astype(np.int)
    df_aux.to_csv('breast-%s-%s.csv' % (''.join(map(str,
                                                    range(1, label + 2))),
                                        ''.join(map(str,
                                                    range(label + 2, 5))),
                                        ), index=False)
    
    
print(df.shape)
