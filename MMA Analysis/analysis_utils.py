import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from fuzzywuzzy import fuzz, process

# Creates a folder to store the plots
os.makedirs('plots', exist_ok=True)

def load_and_clean_data(csv_path='ufc-master.csv'):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.drop_duplicates(inplace=True)
    
    columns_to_drop = [
        "RWFlyweightRank", "RWFeatherweightRank", "RWStrawweightRank", "RWBantamweightRank", "RHeavyweightRank",
        "RLightHeavyweightRank", "RMiddleweightRank", "RWelterweightRank", "RLightweightRank", "RFeatherweightRank",
        "RBantamweightRank", "RFlyweightRank", "RPFPRank", "BWFlyweightRank", "BWFeatherweightRank",
        "BWStrawweightRank", "BWBantamweightRank", "BHeavyweightRank", "BLightHeavyweightRank", "BMiddleweightRank",
        "BWelterweightRank", "BLightweightRank", "BFeatherweightRank", "BBantamweightRank", "BFlyweightRank", 
        "RedDecOdds", "BlueDecOdds", "RSubOdds", "BSubOdds", "RKOOdds", "BKOOdds", "RedExpectedValue", "BlueExpectedValue", 
        "BlueCurrentLoseStreak", "BlueCurrentWinStreak", "BlueDraws", "BlueAvgSigStrLanded", "BlueAvgSigStrPct", "BlueAvgSubAtt",
        "BlueAvgTDLanded", "BlueAvgTDPct", "BlueLongestWinStreak", "RedCurrentLoseStreak", "RedCurrentWinStreak", "RedDraws", 
        "RedAvgSigStrLanded", "RedAvgSigStrPct", "RedAvgSubAtt", "RedAvgTDLanded", "RedAvgTDPct", "RedLongestWinStreak",
        "LoseStreakDif", "WinStreakDif", "LongestWinStreakDif", "EmptyArena"
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df

def finishes(df):
    df_finishes = df[['RedFighter', 'BlueFighter', 'Date', 'Finish', 'WeightClass', 'Gender', 'Winner']].copy()
    df_finishes['Year'] = pd.to_datetime(df_finishes['Date']).dt.year

    # compiling decision methods
    df_finishes['Finish'] = df_finishes['Finish'].replace({
        'M-DEC': 'DEC',
        'S-DEC': 'DEC',
        'U-DEC': 'DEC'
    })

    # overall counts
    counts = df_finishes['Finish'].value_counts()
    print(f"Number of finishes by type (2010 - 2024):\n{counts}\n")

    total_by_year = df_finishes.groupby('Year').size().rename('Total')

    finishes_by_year = df_finishes.groupby(['Year', 'Finish']).size().reset_index(name='Count')

    finishes_by_year = finishes_by_year.merge(total_by_year, on='Year')
    finishes_by_year['Ratio'] = finishes_by_year['Count'] / finishes_by_year['Total']

    
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(12, 10))
    sns.lineplot(
        data=finishes_by_year,
        x='Year',
        y='Ratio',
        hue='Finish',
        marker='o'
    )

    plt.title("Finish Type Ratios Over Time")
    plt.xlabel("Year")
    plt.ylabel("Finish Ratio")
    plt.ylim(0, .6)
    plt.yticks(np.arange(0, 0.65, 0.05))
    plt.legend(title='Finish Type')
    plt.tight_layout()
    plt.savefig('plots/finish_ratios.png')
    
    return finishes_by_year

def analyze_age_and_experience(df):
    sns.set_style("whitegrid")  

    df_age = df[['RedFighter', 'BlueFighter', 'WeightClass', 'RedWins', 'RedLosses', 
                 'BlueWins', 'BlueLosses', 'Gender', 'RedAge', 'BlueAge', 'AgeDif', 'Winner']].copy()
    
    df_age['Older Fighter Wins'] = ((df_age['Winner'] == 'Blue') & (df_age['AgeDif'] < 0)) | \
                                   ((df_age['Winner'] == 'Red') & (df_age['AgeDif'] > 0))
    
    df_age['TotalRedFights'] = df_age['RedWins'] + df_age['RedLosses'] 
    df_age['TotalBlueFights'] = df_age['BlueWins'] + df_age['BlueLosses'] 
    
    df_age['More Experienced Wins'] = ((df_age['Winner'] == 'Blue') & (df_age['TotalBlueFights'] > df_age['TotalRedFights'])) | \
                                      ((df_age['Winner'] == 'Red') & (df_age['TotalRedFights'] > df_age['TotalBlueFights']))
    
    older_wins_percentage = df_age['Older Fighter Wins'].mean()
    more_exp_percentage = df_age['More Experienced Wins'].mean()
    
    by_weight_class = df_age.groupby('WeightClass')[['Older Fighter Wins', 'More Experienced Wins']].mean() * 100
    by_weight_class = by_weight_class.round(2)
    
    print(f"Overall percentage of fights where the older fighter wins: {(older_wins_percentage * 100).round(2)}%")
    print(f"Overall percentage of fights where the more experienced fighter wins (more fights): {(more_exp_percentage * 100).round(2)}%")
    print("\n")
    
    # Plot for Age
    plt.figure(figsize=(10, 6))
    plt.bar(by_weight_class.index, by_weight_class['Older Fighter Wins'], color='skyblue')
    plt.title('Win % of Older Fighter Across Weight Classes')
    plt.ylabel('Win Percentage (%)')
    plt.xlabel('Weight Class')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('plots/age_analysis.png')
    plt.close()
    
    # Plot for Experience
    plt.figure(figsize=(10, 6))
    plt.bar(by_weight_class.index, by_weight_class['More Experienced Wins'], color='salmon')
    plt.title('Win % of More Experienced Fighter Across Weight Classes')
    plt.ylabel('Win Percentage (%)')
    plt.xlabel('Weight Class')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('plots/experience_analysis.png')
    plt.close()
    
    return by_weight_class

def analyze_stance(df):
    df['RedStance'] = df['RedStance'].str.strip()
    df['BlueStance'] = df['BlueStance'].str.strip()
    
    df['WinnerStance'] = np.where(df['Winner'] == 'Red', df['RedStance'], df['BlueStance'])
    df['LoserStance'] = np.where(df['Winner'] == 'Red', df['BlueStance'], df['RedStance'])
    
    wins_df = df['WinnerStance'].value_counts().reset_index()
    wins_df.columns = ['Stance', 'Wins']
    
    losses_df = df['LoserStance'].value_counts().reset_index()
    losses_df.columns = ['Stance', 'Losses']
    
    stance_stats = pd.merge(wins_df, losses_df, on='Stance', how='outer').fillna(0)
    stance_stats[['Wins', 'Losses']] = stance_stats[['Wins', 'Losses']].astype(int)
    stance_stats['TotalFights'] = stance_stats['Wins'] + stance_stats['Losses']
    stance_stats = stance_stats[stance_stats['TotalFights'] >= 10].copy()
    stance_stats['Win Rate'] = ((stance_stats['Wins'] / stance_stats['TotalFights']) * 100).round(1)
    
    # Specifically checking orthodox vs southpaw 
    orthodox_vs_southpaw = df[((df['RedStance'] == 'Orthodox') & (df['BlueStance'] == 'Southpaw')) | 
                              ((df['RedStance'] == 'Southpaw') & (df['BlueStance'] == 'Orthodox'))]
    
    orthodox_wins = orthodox_vs_southpaw[
        ((orthodox_vs_southpaw['RedStance'] == 'Orthodox') & (orthodox_vs_southpaw['Winner'] == 'Red')) |
        ((orthodox_vs_southpaw['BlueStance'] == 'Orthodox') & (orthodox_vs_southpaw['Winner'] == 'Blue'))
    ].shape[0]
    
    total_orthodox_vs_southpaw = orthodox_vs_southpaw.shape[0]
    orthodox_vs_southpaw_winrate = (round(orthodox_wins / total_orthodox_vs_southpaw, 3) * 100) if total_orthodox_vs_southpaw > 0 else 0
    
    new_row = pd.DataFrame({
        'Stance': ['Orthodox vs Southpaw'],
        'Wins': [orthodox_wins],
        'Losses': [total_orthodox_vs_southpaw - orthodox_wins],
        'Win Rate': [orthodox_vs_southpaw_winrate]
    })
    
    stance_stats = pd.concat([stance_stats, new_row], ignore_index=True)
    stance_stats = stance_stats.drop(columns='TotalFights').sort_values(by='Win Rate', ascending=False)
    
    print(stance_stats)
    print("\n")
    
   
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=stance_stats, x='Stance', y='Win Rate', hue='Stance', palette='coolwarm', dodge=False)

    # Remove legend if present (since each stance is already labeled on the x-axis)
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',  
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),  
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=10, color='black')

    plt.title('Win Rate by Fighter Stance')
    plt.ylabel('Win Rate (%)')
    plt.xlabel('Stance')
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 25))  
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('plots/stance_analysis_vertical.png')
    plt.close()
    
    return stance_stats

def analyze_physical_stats(df):
    df_physicalStats = df[['RedFighter', 'BlueFighter', 'WeightClass', 'RedReachCms', 'BlueReachCms',
                           'RedHeightCms', 'BlueHeightCms', 'RedWeightLbs', 'BlueWeightLbs', 'HeightDif', 'ReachDif', 'Winner']].copy()
    
    df_physicalStats['Reach Advantage Wins'] = ((df_physicalStats['Winner'] == 'Blue') & (df_physicalStats['ReachDif'] > 0)) | \
                                               ((df_physicalStats['Winner'] == 'Red') & (df_physicalStats['ReachDif'] < 0))
    
    df_physicalStats['Height Advantage Wins'] = ((df_physicalStats['Winner'] == 'Blue') & (df_physicalStats['HeightDif'] > 0)) | \
                                                ((df_physicalStats['Winner'] == 'Red') & (df_physicalStats['HeightDif'] < 0))
    
    reach_advantage_percent = df_physicalStats['Reach Advantage Wins'].mean()
    print(f"Percentage of fights where the fighter with the longer reach wins: {round(reach_advantage_percent * 100, 2)}%")
    
    height_advantage_percent = df_physicalStats['Height Advantage Wins'].mean()
    print(f"Percentage of fights where the fighter who is taller wins: {round(height_advantage_percent * 100, 2)}%")
    
    # Weight advantage analysis for heavyweights
    df_heavyweights = df_physicalStats[(df_physicalStats['WeightClass'] == 'Heavyweight') &
                                       (df_physicalStats['RedWeightLbs'] > 205) & 
                                       (df_physicalStats['BlueWeightLbs'] > 205)].copy()
    
    df_heavyweights['WeightDif'] = df_heavyweights['BlueWeightLbs'] - df_heavyweights['RedWeightLbs']
    df_heavyweights['WeightAdvantageWins'] = (
        ((df_heavyweights['Winner'] == 'Blue') & (df_heavyweights['WeightDif'] > 0)) |
        ((df_heavyweights['Winner'] == 'Red') & (df_heavyweights['WeightDif'] < 0))
    ).astype(int) 

    weight_advantage_percent = df_heavyweights['WeightAdvantageWins'].mean()
    print(f"Percentage of heavyweight fights where the heavier fighter wins: {round(weight_advantage_percent * 100, 2)}%\n")

    # doing a quick 1 sample t test
    t_stat, p_val = stats.ttest_1samp(df_heavyweights['WeightAdvantageWins'], 0.5)

    print("T-test: Is the heavier fighter more likely to win in the heavyweight division?")
    print(f"  t = {t_stat:.4f}, p = {p_val:.4f}")
    
    # getting the absolute value because if red fighter is heavier the difference will be negitive
    df_heavyweights['AbsWeightGap'] = df_heavyweights['WeightDif'].abs()

    # Run OLS regression
    model = smf.ols(formula="WeightAdvantageWins ~ AbsWeightGap", data=df_heavyweights).fit()
    print(model.summary())

    # Binning for visual analysis
    labels = ['0–5', '5–10', '10–15', '15–20', '20–25', '25–30', '30–60']
    df_heavyweights['WeightGapBin'] = pd.cut(
        df_heavyweights['AbsWeightGap'],
        bins=[0, 5, 10, 15, 20, 25, 30, 60],
        labels=labels
    )

    
    bin_stats = df_heavyweights.groupby('WeightGapBin', observed=True)['WeightAdvantageWins'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=bin_stats, x='WeightGapBin', y='WeightAdvantageWins', color='steelblue')
    plt.title('Heavier Fighter Win Rate by Weight Gap (Heavyweight Only)')
    plt.ylabel('Win Rate')
    plt.xlabel('Weight Difference (lbs)')
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1, 0.05))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('plots/heavyweight_gap_bins.png')

    # Reach and Height Advantage by weight class
    print("\n=== Reach and Height Advantage Win Rates by Weight Class ===")
    weight_classes = df_physicalStats['WeightClass'].dropna().unique()
    class_stats = []
    for wc in sorted(weight_classes):
        wc_df = df_physicalStats[df_physicalStats['WeightClass'] == wc]
        reach_winrate_wc = wc_df['Reach Advantage Wins'].mean()
        height_winrate_wc = wc_df['Height Advantage Wins'].mean()
        class_stats.append({
            'WeightClass': wc,
            'ReachAdvWinRate': reach_winrate_wc * 100,
            'HeightAdvWinRate': height_winrate_wc * 100
        })
        
        print(f"{wc}:")
        print(f"  Reach Advantage Win Rate: {round(reach_winrate_wc * 100, 2)}%")
        print(f"  Height Advantage Win Rate: {round(height_winrate_wc * 100, 2)}%\n")
        
    stats_df = pd.DataFrame(class_stats)

    weight_order = [
        'Strawweight', 'Flyweight', 'Bantamweight', 'Featherweight',
        'Lightweight', 'Welterweight', 'Middleweight', 'Light Heavyweight', 'Heavyweight'
    ]
    
    stats_df['WeightOrder'] = stats_df['WeightClass'].apply(lambda x: weight_order.index(x) if x in weight_order else -1)
    stats_df = stats_df.sort_values('WeightOrder')

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=stats_df, x='WeightClass', y='ReachAdvWinRate', marker='o', label='Reach Advantage')
    sns.lineplot(data=stats_df, x='WeightClass', y='HeightAdvWinRate', marker='s', label='Height Advantage')
    plt.title('Advantage Win Rates by Weight Class')
    plt.ylabel('Win Rate (%)')
    plt.xlabel('Weight Class')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/advantage_by_weightclass.png')
    plt.close()

    return (reach_advantage_percent, height_advantage_percent, weight_advantage_percent)

def compile_fighter_stats(df):
    red_stats = df[['RedFighter', 'RedWins', 'RedLosses', 'RedStance', 'WeightClass', 'Gender',
        'RedTotalRoundsFought', 'RedTotalTitleBouts', 'RedWinsByKO', 'RedWinsBySubmission', 'RedWinsByTKODoctorStoppage',
        'RedHeightCms', 'RedReachCms', 'RedWeightLbs', 'RedAge']].copy()
    
    blue_stats = df[['BlueFighter', 'BlueWins', 'BlueLosses', 'BlueStance', 'WeightClass', 'Gender',
        'BlueTotalRoundsFought', 'BlueTotalTitleBouts', 'BlueWinsByKO', 'BlueWinsBySubmission', 'BlueWinsByTKODoctorStoppage',
        'BlueHeightCms', 'BlueReachCms', 'BlueWeightLbs', 'BlueAge']].copy()
    
    red_stats.columns = ['Fighter', 'Wins', 'Losses', 'Stance', 'Weight Class', 'Gender', 'Total Rounds Fought', 'Title fights', 'KOs',
                         'Submissions', 'TKO/DoctorStoppage', 'Height', 'Reach', 'Weight', 'age']
    
    blue_stats.columns = ['Fighter', 'Wins', 'Losses', 'Stance', 'Weight Class', 'Gender', 'Total Rounds Fought', 'Title fights', 'KOs',
                         'Submissions', 'TKO/DoctorStoppage', 'Height', 'Reach', 'Weight', 'age']
    
    all_fighters = pd.concat([red_stats, blue_stats], ignore_index=True)

    # cleaning data
    all_fighters['Stance'] = all_fighters['Stance'].str.strip()
    all_fighters['Fighter'] = all_fighters['Fighter'].str.strip().str.title()

    # using fuzzywuzzy to handle spelling errors in fighter names
    unique_names = all_fighters['Fighter'].unique()
    name_map = {}

    for name in unique_names:
        if name in name_map:
            continue

    all_fighters['Fighter'] = all_fighters['Fighter'].replace(name_map)
    
    matches = process.extract(name, unique_names, scorer=fuzz.token_sort_ratio, limit=10)
    for match_name, score in matches:
        if score >= 90 and match_name != name:
            name_map[match_name] = name
    
    
    fighter_stats = all_fighters.groupby('Fighter', as_index=False).agg({
        'Wins': 'max',
        'Losses': 'max',
        'Stance': 'first',
        'Weight Class': 'first',
        'Total Rounds Fought': 'max', 
        'Title fights': 'max', 
        'KOs': 'max',             
        'Submissions': 'max', 
        'TKO/DoctorStoppage': 'max',   
        'age': 'max',
        'Gender': 'first',
        'Height': 'first',
        'Weight':'first',
        'Reach': 'first'
    })
    
    fighter_stats['WinRate'] = fighter_stats['Wins'] / (fighter_stats['Wins'] + fighter_stats['Losses'])
    fighter_stats['WinRate'] = fighter_stats['WinRate'].round(3)
    
    # Fixing the one outlier for total fight rounds
    fighter_name = 'Brian Kelleher'
    fighter_stats.loc[fighter_stats['Fighter'] == fighter_name, 'Total Rounds Fought'] = (
        fighter_stats.loc[fighter_stats['Fighter'] == fighter_name, 'Wins'] +
        fighter_stats.loc[fighter_stats['Fighter'] == fighter_name, 'Losses']
    ) * 3 
    
    fighter_stats = fighter_stats.sort_values(by='Total Rounds Fought', ascending=False).reset_index(drop=True)
    fighter_stats.to_csv('fighter_stats_cleaned.csv', index=False)
    fighter_stats
    
    return fighter_stats


def analyze_win_method(df):
    #create figure for win method distribution
    df_temp = df[['Finish']].dropna().copy()

    df_temp['Finish'] = df_temp['Finish'].replace({
        'M-DEC': 'DEC',
        'S-DEC': 'DEC',
        'U-DEC': 'DEC'
    })
    
    fixed_order = ["KO/TKO", "DQ", "DEC", "Overturned", "SUB"]
    method_counts = df_temp['Finish'].value_counts().reindex(fixed_order).fillna(0)
    explode = [0.0 if lab not in ["DQ", "Overturned"] else 0.1 for lab in fixed_order]
    total = method_counts.sum()
    def autopct_func(pct):
        count = int(round(pct * total / 100.0))
        return f"{count} ({pct:.1f}%)"
    plt.figure(figsize=(6,6))
    plt.pie(method_counts.values,
            labels=method_counts.index,
            autopct=autopct_func,
            startangle=140,
            explode=explode,
            labeldistance=1.0,
            pctdistance=0.7)
    plt.title("Overall Distribution of Win Methods")
    plt.tight_layout()
    plt.savefig('plots/win_method_distribution.png')
    plt.close()
    
    return method_counts

def analyze_rounds(df):
    # Create a new DataFrame for round analysis

    round_counts = df['FinishRound'].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=round_counts.index.astype(str), y=round_counts.values, palette="muted")

    plt.title("Distribution of Finish Round Ended")
    plt.xlabel("Round")
    plt.ylabel("Number of Fights")
    plt.tight_layout()
    plt.savefig("plots/round_distribution.png")
    plt.close()

    print(round_counts)

def analyze_gender_difference_all(df):
    red_df = df[['RedFighter', 'Gender', 'RedAge', 'RedHeightCms', 'RedReachCms', 'RedWins']].copy()
    red_df = red_df.rename(columns={
        'RedFighter': 'Fighter',
        'RedAge': 'Age',
        'RedHeightCms': 'Height',
        'RedReachCms': 'Reach',
        'RedWins': 'Wins'
    })
    
    blue_df = df[['BlueFighter', 'Gender', 'BlueAge', 'BlueHeightCms', 'BlueReachCms', 'BlueWins']].copy()
    blue_df = blue_df.rename(columns={
        'BlueFighter': 'Fighter',
        'BlueAge': 'Age',
        'BlueHeightCms': 'Height',
        'BlueReachCms': 'Reach',
        'BlueWins': 'Wins'
    })
    
    # combine red and blue data
    fighter_df = pd.concat([red_df, blue_df], ignore_index=True)
    
    fighter_df = fighter_df.dropna(subset=['Gender', 'Age', 'Height', 'Reach', 'Wins'])
    
    gender_cols = ['Age', 'Height', 'Reach', 'Wins']
    for col in gender_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='Gender', y=col, data=fighter_df)
        plt.title(f"Gender vs. {col}")
        plt.tight_layout()
        plt.savefig(f"plots/gender_all_{col}_boxplot.png")
        plt.close()
    
def analyze_fight_timeline(df):
    #create a figure for fight timeline every year

    df['Year'] = pd.to_datetime(df['Date']).dt.year
    year_counts = df['Year'].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=year_counts.index, y=year_counts.values, marker='o')
    plt.title("Number of Fights per Year")
    plt.xlabel("Year")
    plt.ylabel("Count of Fights")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/fight_count_by_year.png")
    plt.close()

    print(year_counts)