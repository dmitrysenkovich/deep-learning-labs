import pandas as pd

df = pd.read_csv("submission_vgg16_vertical_flip.csv")
df['label'] = df['label'].clip(lower=0.01, upper=0.99)

print(df.head())

df.to_csv('asd.csv', index=False)
