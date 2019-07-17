import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import os


df = pd.read_excel('data/PH2/original.xlsx')
df.sort_values(by='Image Name', inplace=True)

img_path = 'data/PH2/PH2 Dataset images'
img_out_path = 'data/PH2/imgs'

if not os.path.exists(img_out_path):
    os.makedirs(img_out_path)

# FIXME: remove this comment
for i in list(os.walk(img_path))[0][1]:
    next_path = os.path.join(img_path, i, i + '_Dermoscopic_Image',
                             i + '.bmp')
    shutil.copyfile(next_path, os.path.join(img_out_path, i + '.bmp'))

common_nevus = ~pd.isnull(df['Common Nevus'].values)
atypical_nevus = ~pd.isnull(df['Atypical Nevus'].values)
melanoma = ~pd.isnull(df['Melanoma'].values)
diagnosis = 0 * common_nevus + 1 * atypical_nevus + 2 * melanoma

pigment = df['Pigment Network\n(AT/T)'].values
pigment = (pigment == 'AT').astype(np.int)

colors_of_interest = ['White', 'Red', 'Light-Brown', 'Dark-Brown',
                      'Blue-Gray', 'Black']
colors = (df[colors_of_interest] == 'X').astype(np.int)

dots = df['Dots/Globules\n(A/AT/T)'].values
dots = np.stack((dots == 'A',
                 dots == 'AT',
                 dots == 'T'), axis=1)

blue = df['Blue-Whitish Veil\n(A/P)'] == 'P'
regression = df['Blue-Whitish Veil\n(A/P)'] == 'P'
streaks = df['Streaks\n(A/P)'] == 'P'

df_final = np.hstack((df['Asymmetry\n(0/1/2)'].values[:, np.newaxis],
                      pigment[:, np.newaxis],
                      dots,
                      streaks[:, np.newaxis],
                      regression[:, np.newaxis],
                      blue[:, np.newaxis],
                      colors.as_matrix(),
                      diagnosis[:, np.newaxis]))

df_final = pd.DataFrame(df_final,
                        columns=['Asymmetry', 'Pigment',
                                 'Dots-Absent', 'Dots-Typical',
                                 'Dots-Atypical', 'Streaks', 'Regression',
                                 'BlueVeil'] + colors_of_interest +
                        ['Diagnosis'])

df_0_12 = df_final.copy()
df_0_12['Diagnosis'] = (df_final.Diagnosis.values > 0).astype(int)
df_0_12.to_csv('data/PH2/ph2-0-12.csv', index=False)

df_01_2 = df_final.copy()
df_01_2['Diagnosis'] = (df_final.Diagnosis.values > 1).astype(int)
df_01_2.to_csv('data/PH2/ph2-01-2.csv', index=False)
