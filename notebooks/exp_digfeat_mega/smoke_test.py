import sys
sys.path.insert(0, "notebooks/exp_digfeat_mega")
import pandas as pd
import numpy as np
from experiment_digfeat_mega import add_features

df = pd.DataFrame({
    'age': [17.5, 22.0, np.nan],
    'daily_screen_time_hours': [3.456, 7.0, 1.2],
    'sleep_hours': [6.5, 8.0, 7.25],
    'weekend_screen_time': [4.0, 5.0, 2.0],
    'notifications_per_day': [50, 60, 40],
    'app_opens_per_day': [30, 35, 20],
    'social_media_hours': [1.5, 2.0, 1.0],
    'gaming_hours': [1.0, 1.5, 0.5],
    'work_study_hours': [6.0, 7.0, 8.0],
    'gender': ['Male', 'Female', 'Other'],
    'stress_level': ['Low', 'High', np.nan],
    'academic_work_impact': ['Yes', 'No', 'Unknown'],
})
num_cols = [c for c in df.columns if c not in ('gender', 'stress_level', 'academic_work_impact')]
out = add_features(df.copy(), num_cols)
r = out.iloc[0]
assert abs(r['daily_screen_time_hours_frac'] - 0.456) < 1e-9
assert r['daily_screen_time_hours_d1'] == 4
assert r['sleep_hours_is_int'] == 0 and out.iloc[1]['sleep_hours_is_int'] == 1
nanrow = out.iloc[2]
assert nanrow['age_frac'] == -1 and nanrow['age_d1'] == -1 and nanrow['age_is_int'] == -1
assert abs(r['total_screen_time'] - 7.456) < 1e-9
assert out['stress_level'].tolist() == [0, 2, 1]      # Low, High, Unknown->1
assert 'gender' in out.columns                        # gender stage1 me map hota hai
print("ADD_FEATURES SMOKE TEST: ALL PASS")
