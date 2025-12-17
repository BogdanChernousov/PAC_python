import pandas as pd

wells = pd.read_csv('wells_info.csv')

wells['SpudDate'] = pd.to_datetime(wells['SpudDate'])                            # дата начала бурения
# wells['FirstProductionDate'] = pd.to_datetime(wells['FirstProductionDate'])    # дата начала добычи

wells['CompletionDate'] = pd.to_datetime(wells['CompletionDate'])

wells['months_duration'] = ((wells['CompletionDate'] - wells['SpudDate']).dt.days / 30.4).astype(int)
# wells['months_duration'] = ((wells['CompletionDate'] - wells['FirstProductionDate']).dt.days / 30.44).astype(int)

print(wells[['API', 'operatorNameIHS', 'months_duration']].head())