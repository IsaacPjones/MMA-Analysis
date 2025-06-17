import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def mma_fight_analysis(df):
    # 1 if red wins, 0 if blue wins
    df['WinnerBinary'] = (df['Winner'] == 'Red').astype(int)

    df['ReachDiff'] = df['RedReachCms'] - df['BlueReachCms']
    df['HeightDiff'] = df['RedHeightCms'] - df['BlueHeightCms']
    df['AgeDiff'] = df['RedAge'] - df['BlueAge']

    df['RedFights'] = df['RedWins'] + df['RedLosses']
    df['BlueFights'] = df['BlueWins'] + df['BlueLosses']
    df['ExpDiff'] = df['RedFights'] - df['BlueFights']

    df['RedStance'] = df['RedStance'].str.strip()
    df['BlueStance'] = df['BlueStance'].str.strip()
    df['StanceMatchup'] = df['RedStance'] + '_vs_' + df['BlueStance']

    model = smf.ols(formula="WinnerBinary ~ ReachDiff + HeightDiff + AgeDiff + ExpDiff + C(StanceMatchup)", data=df).fit()
    print("OLS Regression Summary:")
    print(model.summary())
    print("\n")

    # Advantage values depending on winner
    df['ReachAdvWinner'] = np.where(df['WinnerBinary'] == 1,
                                    df['RedReachCms'] - df['BlueReachCms'],
                                    df['BlueReachCms'] - df['RedReachCms'])

    df['HeightAdvWinner'] = np.where(df['WinnerBinary'] == 1,
                                     df['RedHeightCms'] - df['BlueHeightCms'],
                                     df['BlueHeightCms'] - df['RedHeightCms'])

    df['AgeAdvWinner'] = np.where(df['WinnerBinary'] == 1,
                                  df['RedAge'] - df['BlueAge'],
                                  df['BlueAge'] - df['RedAge'])

    df['ExpAdvWinner'] = np.where(df['WinnerBinary'] == 1,
                                  df['RedFights'] - df['BlueFights'],
                                  df['BlueFights'] - df['RedFights'])

    # Mean advantages
    print(f"Average Reach Advantage of Winner: {df['ReachAdvWinner'].mean():.2f} cm")
    print(f"Average Height Advantage of Winner: {df['HeightAdvWinner'].mean():.2f} cm")
    print(f"Average Age Advantage of Winner: {df['AgeAdvWinner'].mean():.2f} years")
    print(f"Average Experience Advantage of Winner: {df['ExpAdvWinner'].mean():.2f} fights")

    # 1 sample t-tests against zero (no advantage)
    print("\nT-tests (advantage > 0):")
    print(f"Reach Advantage: t = {stats.ttest_1samp(df['ReachAdvWinner'], 0).statistic:.4f}, "
          f"p = {stats.ttest_1samp(df['ReachAdvWinner'], 0).pvalue:.4f}")
    print(f"Height Advantage: t = {stats.ttest_1samp(df['HeightAdvWinner'], 0).statistic:.4f}, "
          f"p = {stats.ttest_1samp(df['HeightAdvWinner'], 0).pvalue:.4f}")
    print(f"Age Advantage: t = {stats.ttest_1samp(df['AgeAdvWinner'], 0).statistic:.4f}, "
          f"p = {stats.ttest_1samp(df['AgeAdvWinner'], 0).pvalue:.4f}")
    print(f"Experience Advantage: t = {stats.ttest_1samp(df['ExpAdvWinner'], 0).statistic:.4f}, "
          f"p = {stats.ttest_1samp(df['ExpAdvWinner'], 0).pvalue:.4f}")
