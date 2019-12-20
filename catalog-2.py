import pandas as pd
from bpimagelist import parse_bpimagelist
import numba
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.expand_frame_repr', False)

img_df, frag_df = parse_bpimagelist('8.1', r"...")

@numba.jit
def frag_size(kb, r):
    return kb * 1024 + r

@numba.jit
def ret_lvl_frag(lvl):
    return lvl - 65536 if lvl >= 65536 else lvl


@numba.jit(nopython=True)
def ret_days(im_bp_ts, im_ex_ts, fr_bp_ts, fr_ex_ts, fr_ex2_ts, frag_num, n_copy):
    fr_ex_ts = fr_ex2_ts if fr_ex_ts == 0 else fr_ex_ts
    fr_ex_ts = im_ex_ts if fr_ex_ts == 0 else fr_ex_ts
    fr_bp_ts = im_bp_ts if fr_bp_ts == 0 else fr_bp_ts
    if n_copy == 1:
        fr_ex_ts = im_ex_ts
        fr_bp_ts = im_bp_ts
    if frag_num == 1:
        if fr_ex_ts == 2147483647:
            days = 999999
        else:
            days = (fr_ex_ts - fr_bp_ts) / 60 / 60 / 24
    else:
        days = -1
    return days


@numba.jit(nopython=True)
def copy_exp(im_ex_ts, fr_ex_ts, fr_ex2_ts, frag_num, n_copy):
    fr_ex_ts = fr_ex2_ts if fr_ex_ts == 0 else fr_ex_ts
    fr_ex_ts = im_ex_ts if fr_ex_ts == 0 else fr_ex_ts
    if n_copy == '1':
        fr_ex_ts = im_ex_ts
    if frag_num == '1':
        fr_ex_ts = -1
    return fr_ex_ts


@numba.jit(nopython=True)
def copy_exp(im_ex_ts, fr_ex_ts, fr_ex2_ts, frag_num, n_copy):
    fr_ex_ts = fr_ex2_ts if fr_ex_ts == 0 else fr_ex_ts
    fr_ex_ts = im_ex_ts if fr_ex_ts == 0 else fr_ex_ts
    if n_copy == '1':
        fr_ex_ts = im_ex_ts
    if frag_num == '1':
        fr_ex_ts = -1
    return fr_ex_ts


frag_df['ret_sec'] = frag_df['expire_time'].astype(int) - frag_df['media_time'].astype(int)
comb_df = frag_df.join(img_df.set_index('bp_image_id'), on='bp_image_id', lsuffix='_fr')
df = comb_df.loc[:,
     ['client', 'policy', 'policy_type', 'schedule', 'sched_type', 'copy_n', 'job_id', 'copy_n_fr', 'ret_lvl_fr',
      'size_kb', 'backup_time', 'expire_time', 'media_id', 'host', 'media_time', 'ret_lvl',
      'size_kb_fr', 'remainder_b', 'media_descr', 'expire_time_fr', 'frag_n_fr', 'bp_image_id', 'try_to_keep_time',
      'copy_time', 'slp_name', 'slp_done']]


for int_f in ['backup_time', 'expire_time', 'copy_time', 'try_to_keep_time', 'expire_time_fr', 'frag_n_fr', 'copy_n',
              'ret_lvl_fr', 'copy_n_fr', 'size_kb_fr', 'remainder_b', 'copy_n_fr', 'ret_lvl_fr', 'slp_done', 'size_kb' ]:
    print(int_f)
    df[int_f] = pd.to_numeric(df[int_f])

df['real_ret_lvl_fr'] = df['ret_lvl_fr'].apply(ret_lvl_frag).astype(int)

df['Retention days'] = df.apply(
    lambda x: ret_days(x['backup_time'], x['expire_time'], x['copy_time'], x['try_to_keep_time'], x['expire_time_fr'],
                       x['frag_n_fr'], x['copy_n']), axis=1).astype(int)

df['Copy Expiration'] = df.apply(
    lambda x: copy_exp(x['expire_time'], x['try_to_keep_time'], x['expire_time_fr'],
                       x['frag_n_fr'], x['copy_n']), axis=1).astype(int)

df['Media type'] = df['media_descr'].apply(lambda x: x.split(';')[1] if ';' in x else 'other')
df['Storage server'] = df['media_descr'].apply(lambda x: x.split(';')[2] if ';' in x else 'other')
df['media_disk_pool'] = df['media_descr'].apply(lambda x: x.split(';')[3] if ';' in x else 'other')

df.loc[:, ['Retention days', 'real_ret_lvl_fr']].groupby('real_ret_lvl_fr').agg(set)

df['frag_size'] = df.apply(lambda x: frag_size(x['size_kb_fr'], x['remainder_b']), axis=1).astype('int64')

df2 = df.loc[:, ['client', 'policy', 'Media type', 'media_disk_pool', 'schedule', 'host', 'sched_type', 'frag_size',
                 'copy_n_fr', 'Copy Expiration', 'backup_time', 'frag_n_fr']]

z = pd.get_dummies(df2, prefix_sep="__", columns=['client', 'policy'])


grp_df = df2.groupby(
    ['client', 'policy', 'Media type', 'media_disk_pool', 'host', 'schedule', 'sched_type', 'copy_n_fr']) \
    .agg({'Copy Expiration': max, 'frag_size': sum}).reset_index()

from itertools import chain

periods = list(chain(range(1, 14), range(14, 60, 4), range(61, 120, 7), range(120, 360, 14), range(360, 720, 30),
                     range(720, 3600, 360)))
latest_backup_time = df['backup_time'].max()


@numba.jit
def split_expiration(exp):
    seconds = exp - latest_backup_time
    days = seconds // 86400
    for p in periods:
        if days <= p:
            return p
    return 3600


grp_df["Expire Period"] = grp_df['Copy Expiration'].apply(split_expiration)

media_type_grp_df = grp_df.loc[:, ['Media type', "Expire Period", 'frag_size']].groupby(
    ['Media type', "Expire Period"]).agg(sum)

labels = []

for i in df.index:
    df.at[i, 'frag_size'] = 1
for med_type, sub_df in media_type_grp_df.groupby(level=0, as_index=False):
    agg_size = 0
    for p, sizes in sub_df.groupby(level=1, as_index=False):
        agg_size += sizes['frag_size']
        sizes['agg_size'] = agg_size

z = df.loc[:,
    ['client', 'policy', 'slp_name', 'schedule', 'slp_done', 'media_disk_pool', 'host', 'copy_n', 'copy_n_fr',
     'size_kb']]. \
    groupby(['copy_n', 'slp_name', 'slp_done', 'media_disk_pool', 'client', 'host', 'policy', 'schedule']). \
    agg({'copy_n_fr': max, 'size_kb': sum}).reset_index()

x = z.loc[:,
    ['slp_name', 'slp_done', 'media_disk_pool', 'copy_n', 'copy_n_fr', 'size_kb', 'host', 'policy', 'schedule']]. \
    groupby(['copy_n', 'slp_name', 'slp_done', 'media_disk_pool', 'policy', 'host', 'schedule']).agg(
    {'copy_n_fr': max, 'size_kb': sum}) \
    .reset_index()

# SLP 3 - done / 0 - no SLP
f = x.where(x['slp_done'].isin([0, 3])).dropna(thresh=6).loc[:,
    ['copy_n', 'copy_n_fr', 'host', 'size_kb', 'policy', 'schedule']]. \
    groupby(['copy_n', 'copy_n_fr', 'policy', 'host', 'schedule']).agg(sum).unstack('copy_n_fr').reset_index() \
    .set_index(['policy', 'schedule', 'host']).fillna(0)

f['range'] = [(max(x[1:4]) - min([z for z in x[1:4] if z > 0] + [max(x[1:4])])) for x in f.get_values()]
f['2 copies std'] = [(sum(x[1:4]) - max(x[1:4] * 2)) for x in f.get_values()]

f.to_csv(r'E:\model_nbucat\copy_stats.csv')
f.where(f['2 copies std'] > 0.01).dropna().sort_values(by='2 copies std', ascending=False).head(20).to_csv(
    r'E:\model_nbucat\top_20_2_cop_no_slp.csv')
f.where(f['range'] > 0.01).dropna().sort_values(by='range', ascending=False).head(20).to_csv(
    r'E:\model_nbucat\top_20_range_no_slp.csv')

grp_df['Size, GB'] = grp_df['frag_size'].apply(lambda x: x / 1024 / 1024 / 1024)
grp_df.sort_values(by='Size, GB', inplace=True)
sns.set(style="whitegrid")
sns.set(rc={'figure.figsize': (15, 8.27)})
plt.rcParams['patch.linewidth'] = 0

t = grp_df.groupby(['host', 'Media type']).agg(sum).reset_index().sort_values(by='frag_size')
g = sns.barplot(x='host', y='Size, GB', order=t['host'].sort_values().unique(),
                hue_order=['DataDomain', 'AdvancedDisk', 'other'], hue="Media type", data=t)
g.set_xticklabels(g.get_xticklabels(), rotation=75)
plt.legend(loc='upper left')
plt.show()

t = grp_df.groupby(['host', 'copy_n_fr']).agg(sum).reset_index().sort_values(by='frag_size')
g = sns.barplot(x='host', y='Size, GB', order=t['host'].sort_values().unique(), hue_order=['1', '2', '3', '4'],
                hue="copy_n_fr", data=t)
g.set_xticklabels(g.get_xticklabels(), rotation=75)
plt.legend(loc='upper left')
plt.show()

t = grp_df.groupby(['copy_n_fr', 'Media type']).agg(sum).reset_index().sort_values(by='frag_size')
g = sns.barplot(x='Media type', y='Size, GB', hue_order=['1', '2', '3', '4'],
                hue="copy_n_fr", data=t)
plt.legend(loc='upper left')
plt.show()

df2 = df.loc[:, ['media_disk_pool', 'copy_n_fr', 'policy', 'Media type', 'frag_size']]

df2.groupby(['copy_n_fr', 'policy', 'Media type', 'host']).agg(sum).reset_index().sort_values(by='frag_size')
df_ret_pol = df2.groupby(['Retention days', 'policy', 'Media type', 'copy_n_fr']).agg(sum).reset_index().sort_values(
    by='frag_size')
df_r = df_ret_pol.groupby(['Retention days', 'Media type', 'copy_n_fr']).agg(sum).reset_index().sort_values(
    by='frag_size')

t = grp_df.groupby(['policy', 'copy_n_fr']).agg(sum).reset_index().sort_values(by='frag_size')
g = sns.barplot(x='host', y='Size, GB', order=t['host'].sort_values().unique(), hue_order=['1', '2', '3', '4'],
                hue="copy_n_fr", data=t)
g.set_xticklabels(g.get_xticklabels(), rotation=75)
plt.legend(loc='upper left')
plt.show()
